using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

/// <summary>
/// Unity中使用ONNX模型的推理组件（修复版）
/// </summary>
public class OnnxModelInference : MonoBehaviour
{
    [Header("模型设置")]
    [Tooltip("ONNX模型文件路径（相对于StreamingAssets文件夹）")]
    public string modelFileName = "pytorch_model_v2.onnx";

    [Tooltip("是否在Start时自动加载模型")]
    public bool autoLoadOnStart = true;

    [Tooltip("是否使用GPU加速（仅Windows Editor有效）")]
    public bool useGpu = false;

    [Header("性能设置")]
    [Tooltip("推理线程数")]
    public int inferenceThreads = 4;

    [Header("调试信息")]
    public bool showDebugInfo = true;
    public float lastInferenceTime = 0f;

    private InferenceSession _session;
    private string _inputName;
    private string _outputName;
    private bool _isModelLoaded = false;

    public bool IsModelLoaded => _isModelLoaded;

    void Start()
    {
        if (autoLoadOnStart)
        {
            LoadModel();
        }
    }

    public bool LoadModel()
    {
        try
        {
            string modelPath = System.IO.Path.Combine(Application.streamingAssetsPath, modelFileName);
            if (!System.IO.File.Exists(modelPath))
            {
                Debug.LogError($"模型文件不存在: {modelPath}");
                return false;
            }

            var sessionOptions = new SessionOptions();


            if (useGpu)
            {
                try
                {
                    sessionOptions.AppendExecutionProvider_CUDA(0);
                    Debug.Log("✅ 使用 GPU 加速");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"⚠️ GPU 不可用，回退到 CPU: {ex.Message}");
                }
            }

            sessionOptions.InterOpNumThreads = 1;
            sessionOptions.IntraOpNumThreads = inferenceThreads;
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            _session = new InferenceSession(modelPath, sessionOptions);
            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();

            // 验证输入维度
            var inputMeta = _session.InputMetadata[_inputName];
            if (inputMeta.Dimensions.Length != 3 || inputMeta.Dimensions[1] != 1)
            {
                Debug.LogError("❌ 模型输入维度不匹配！期望 [batch, 1, time]");
                _session.Dispose();
                return false;
            }

            _isModelLoaded = true;

            if (showDebugInfo)
            {
                Debug.Log($"✅ 模型加载成功: {modelFileName}");
                Debug.Log($"输入: {_inputName} {string.Join("x", inputMeta.Dimensions)}");
                Debug.Log($"输出: {_outputName} {string.Join("x", _session.OutputMetadata[_outputName].Dimensions)}");
            }

            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"❌ 加载模型失败: {ex.Message}\n{ex.StackTrace}");
            return false;
        }
    }

    /// <summary>
    /// 执行推理（同步）
    /// </summary>
    public float[] Infer(float[] inputData)
    {
        if (!_isModelLoaded)
        {
            Debug.LogError("❌ 模型未加载！");
            return null;
        }

        if (inputData == null || inputData.Length == 0)
        {
            Debug.LogError("❌ 输入数据为空！");
            return null;
        }

        try
        {
            var startTime = Time.realtimeSinceStartup;

            // 裁剪输入到 [-1, 1]（匹配 AudioClip 范围）
            for (int i = 0; i < inputData.Length; i++)
            {
                inputData[i] = Mathf.Clamp(inputData[i], -1f, 1f);
            }

            var inputDimensions = new[] { 1, 1, inputData.Length };
            var inputTensor = new DenseTensor<float>(inputData, inputDimensions);
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) };

            float[] output;
            using (var results = _session.Run(inputs))
            {
                output = results.First().AsEnumerable<float>().ToArray();
            }

            lastInferenceTime = (Time.realtimeSinceStartup - startTime) * 1000f;

            if (showDebugInfo)
            {
                Debug.Log($"✅ 推理完成: 输入={inputData.Length}, 输出={output.Length}, 耗时={lastInferenceTime:F2}ms");
            }

            return output;
        }
        catch (Exception ex)
        {
            Debug.LogError($"❌ 推理失败: {ex.Message}\n{ex.StackTrace}");
            return null;
        }
    }

    // ⚠️ 移除 InferAsync（ONNX Session 非线程安全）

    public float[] InferFromAudioClip(AudioClip audioClip)
    {
        if (audioClip == null)
        {
            Debug.LogError("❌ AudioClip 为空！");
            return null;
        }

        float[] samples = new float[audioClip.samples * audioClip.channels];
        audioClip.GetData(samples, 0);

        if (audioClip.channels == 2)
        {
            float[] mono = new float[audioClip.samples];
            for (int i = 0; i < audioClip.samples; i++)
            {
                mono[i] = (samples[i * 2] + samples[i * 2 + 1]) / 2f;
            }
            samples = mono;
        }

        return Infer(samples);
    }

    public List<float[]> InferBatch(List<float[]> inputBatch)
    {
        if (!_isModelLoaded)
        {
            Debug.LogError("❌ 模型未加载！");
            return null;
        }

        var results = new List<float[]>();
        foreach (var input in inputBatch)
        {
            var output = Infer(input);
            if (output != null)
            {
                results.Add(output);
            }
        }
        return results;
    }

    void OnDestroy()
    {
        if (_session != null)
        {
            _session.Dispose();
            _session = null;
        }
        _isModelLoaded = false;
    }

    void OnApplicationQuit() => OnDestroy();
}