using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

/// <summary>
/// NovaSR ONNX 模型推理组件（修复版）
/// </summary>
public class OnnxModelInference : MonoBehaviour
{
    [Header("模型设置")]
    [Tooltip("ONNX模型文件路径（相对于StreamingAssets文件夹）")]
    public string modelFileName = "pytorch_model_v2.onnx";

    [Tooltip("是否在Start时自动加载模型")]
    public bool autoLoadOnStart = true;

    [Tooltip("是否使用GPU加速")]
    public bool useGpu = false;

    [Header("音频设置")]
    [Tooltip("输入采样率（应为16000 Hz）")]
    public int inputSampleRate = 16000;

    [Tooltip("输出采样率（应为48000 Hz）")]
    public int outputSampleRate = 48000;

    [Header("性能设置")]
    [Tooltip("推理线程数")]
    public int inferenceThreads = 4;

    [Tooltip("是否启用模型优化")]
    public bool enableOptimization = true;

    [Header("调试信息")]
    public bool showDebugInfo = true;
    public float lastInferenceTime = 0f;
    public int lastInputLength = 0;
    public int lastOutputLength = 0;

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
                Debug.LogError($"❌ 模型文件不存在: {modelPath}");
                return false;
            }

            var sessionOptions = new SessionOptions();

            // GPU 加速
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

            // 性能优化
            sessionOptions.InterOpNumThreads = 1;
            sessionOptions.IntraOpNumThreads = inferenceThreads;

            if (enableOptimization)
            {
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            }

            _session = new InferenceSession(modelPath, sessionOptions);
            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();

            // 验证模型
            var inputMeta = _session.InputMetadata[_inputName];
            var outputMeta = _session.OutputMetadata[_outputName];

            if (inputMeta.Dimensions.Length != 3 || inputMeta.Dimensions[1] != 1)
            {
                Debug.LogError($"❌ 模型输入维度不匹配！期望 [batch, 1, time]，实际 [{string.Join(", ", inputMeta.Dimensions)}]");
                _session.Dispose();
                return false;
            }

            _isModelLoaded = true;

            if (showDebugInfo)
            {
                Debug.Log($"✅ 模型加载成功: {modelFileName}");
                Debug.Log($"   输入: {_inputName} [{string.Join(", ", inputMeta.Dimensions)}]");
                Debug.Log($"   输出: {_outputName} [{string.Join(", ", outputMeta.Dimensions)}]");
                Debug.Log($"   期望采样率: {inputSampleRate} Hz → {outputSampleRate} Hz");
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
    /// 执行推理（同步）- 核心方法
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

            // ✅ 修复1：不要裁剪！AudioClip 数据本身就是 [-1, 1]
            // ❌ 移除这段代码：
            // for (int i = 0; i < inputData.Length; i++)
            // {
            //     inputData[i] = Mathf.Clamp(inputData[i], -1f, 1f);
            // }

            // ✅ 修复2：确保输入数据没有 NaN 或 Inf
            bool hasInvalidData = false;
            for (int i = 0; i < inputData.Length; i++)
            {
                if (float.IsNaN(inputData[i]) || float.IsInfinity(inputData[i]))
                {
                    inputData[i] = 0f;
                    hasInvalidData = true;
                }
            }

            if (hasInvalidData && showDebugInfo)
            {
                Debug.LogWarning("⚠️ 输入数据包含 NaN/Inf，已替换为 0");
            }

            // 创建输入张量
            var inputDimensions = new[] { 1, 1, inputData.Length };
            var inputTensor = new DenseTensor<float>(inputData, inputDimensions);
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) };

            // 执行推理
            float[] output;
            using (var results = _session.Run(inputs))
            {
                output = results.First().AsEnumerable<float>().ToArray();
            }

            lastInferenceTime = (Time.realtimeSinceStartup - startTime) * 1000f;
            lastInputLength = inputData.Length;
            lastOutputLength = output.Length;

            if (showDebugInfo)
            {
                float inputRMS = CalculateRMS(inputData);
                float outputRMS = CalculateRMS(output);
                Debug.Log($"✅ 推理完成:\n" +
                         $"   输入: {inputData.Length} samples, RMS={inputRMS:F4}\n" +
                         $"   输出: {output.Length} samples, RMS={outputRMS:F4}\n" +
                         $"   上采样率: {(float)output.Length / inputData.Length:F2}x\n" +
                         $"   耗时: {lastInferenceTime:F2}ms");
            }

            return output;
        }
        catch (Exception ex)
        {
            Debug.LogError($"❌ 推理失败: {ex.Message}\n{ex.StackTrace}");
            return null;
        }
    }

    /// <summary>
    /// 从 AudioClip 推理（改进版）
    /// </summary>
    public float[] InferFromAudioClip(AudioClip audioClip, bool resampleTo16k = false)
    {
        if (audioClip == null)
        {
            Debug.LogError("❌ AudioClip 为空！");
            return null;
        }

        // 获取原始数据
        float[] samples = new float[audioClip.samples * audioClip.channels];
        audioClip.GetData(samples, 0);

        // ✅ 修复3：正确处理立体声
        if (audioClip.channels == 2)
        {
            float[] mono = new float[audioClip.samples];
            for (int i = 0; i < audioClip.samples; i++)
            {
                // 使用标准的立体声转单声道公式
                mono[i] = (samples[i * 2] + samples[i * 2 + 1]) * 0.5f;
            }
            samples = mono;
        }

        // ✅ 修复4：检查采样率
        if (audioClip.frequency != inputSampleRate)
        {
            if (resampleTo16k)
            {
                Debug.LogWarning($"⚠️ AudioClip 采样率为 {audioClip.frequency} Hz，将重采样到 {inputSampleRate} Hz");
                samples = SimpleResample(samples, audioClip.frequency, inputSampleRate);
            }
            else
            {
                Debug.LogError($"❌ AudioClip 采样率不匹配！期望 {inputSampleRate} Hz，实际 {audioClip.frequency} Hz\n" +
                              $"请设置 resampleTo16k=true 或使用正确采样率的音频");
                return null;
            }
        }

        return Infer(samples);
    }

    /// <summary>
    /// 创建输出 AudioClip
    /// </summary>
    public AudioClip CreateOutputAudioClip(float[] outputData, string clipName = "NovaSR_Output")
    {
        if (outputData == null || outputData.Length == 0)
        {
            Debug.LogError("❌ 输出数据为空！");
            return null;
        }

        // ✅ 修复5：确保输出数据在 [-1, 1] 范围内
        float maxAbs = 0f;
        for (int i = 0; i < outputData.Length; i++)
        {
            float abs = Mathf.Abs(outputData[i]);
            if (abs > maxAbs) maxAbs = abs;
        }

        // 如果超出范围，进行归一化
        if (maxAbs > 1f)
        {
            Debug.LogWarning($"⚠️ 输出数据超出范围 (max={maxAbs:F3})，进行归一化");
            for (int i = 0; i < outputData.Length; i++)
            {
                outputData[i] /= maxAbs;
            }
        }

        AudioClip clip = AudioClip.Create(clipName, outputData.Length, 1, outputSampleRate, false);
        clip.SetData(outputData, 0);
        return clip;
    }

    /// <summary>
    /// 完整的音频超分辨率处理流程
    /// </summary>
    public AudioClip ProcessAudio(AudioClip inputClip, string outputName = "Enhanced_Audio")
    {
        if (!_isModelLoaded)
        {
            Debug.LogError("❌ 模型未加载！");
            return null;
        }

        // 从 AudioClip 推理
        float[] outputData = InferFromAudioClip(inputClip, resampleTo16k: true);
        if (outputData == null) return null;

        // 创建输出 AudioClip
        return CreateOutputAudioClip(outputData, outputName);
    }

    /// <summary>
    /// 批量推理
    /// </summary>
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

    // ========== 辅助方法 ==========

    /// <summary>
    /// 计算 RMS（均方根）用于调试
    /// </summary>
    private float CalculateRMS(float[] samples)
    {
        if (samples == null || samples.Length == 0) return 0f;

        double sum = 0;
        for (int i = 0; i < samples.Length; i++)
        {
            sum += samples[i] * samples[i];
        }
        return (float)Math.Sqrt(sum / samples.Length);
    }

    /// <summary>
    /// 简单的线性重采样（仅用于采样率转换）
    /// </summary>
    private float[] SimpleResample(float[] input, int fromRate, int toRate)
    {
        if (fromRate == toRate) return input;

        double ratio = (double)fromRate / toRate;
        int outputLength = (int)(input.Length / ratio);
        float[] output = new float[outputLength];

        for (int i = 0; i < outputLength; i++)
        {
            double srcIndex = i * ratio;
            int idx1 = (int)srcIndex;
            int idx2 = Math.Min(idx1 + 1, input.Length - 1);
            float frac = (float)(srcIndex - idx1);

            // 线性插值
            output[i] = input[idx1] * (1f - frac) + input[idx2] * frac;
        }

        return output;
    }

    /// <summary>
    /// 验证模型输出质量
    /// </summary>
    public bool ValidateModelOutput(AudioClip testClip)
    {
        Debug.Log("🔍 开始模型验证...");

        float[] output = InferFromAudioClip(testClip, resampleTo16k: true);
        if (output == null)
        {
            Debug.LogError("❌ 验证失败：推理返回 null");
            return false;
        }

        // 检查输出质量
        float rms = CalculateRMS(output);
        float maxAbs = 0f;
        int nanCount = 0;

        for (int i = 0; i < output.Length; i++)
        {
            if (float.IsNaN(output[i]) || float.IsInfinity(output[i]))
            {
                nanCount++;
            }
            float abs = Mathf.Abs(output[i]);
            if (abs > maxAbs) maxAbs = abs;
        }

        Debug.Log($"📊 验证结果:\n" +
                 $"   输出长度: {output.Length}\n" +
                 $"   RMS: {rms:F4}\n" +
                 $"   最大值: {maxAbs:F4}\n" +
                 $"   异常值数量: {nanCount}");

        bool isValid = nanCount == 0 && rms > 0.001f && maxAbs < 100f;
        Debug.Log(isValid ? "✅ 模型验证通过" : "❌ 模型验证失败");
        return isValid;
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