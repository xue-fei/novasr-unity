using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

/// <summary>
/// Unity中使用ONNX模型的推理组件
/// </summary>
public class OnnxModelInference : MonoBehaviour
{
    [Header("模型设置")]
    [Tooltip("ONNX模型文件路径（相对于StreamingAssets文件夹）")]
    public string modelFileName = "pytorch_model_v2.onnx";

    [Tooltip("是否在Start时自动加载模型")]
    public bool autoLoadOnStart = true;

    [Tooltip("是否使用GPU加速（需要安装GPU版本的OnnxRuntime）")]
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

    /// <summary>
    /// 模型是否已加载
    /// </summary>
    public bool IsModelLoaded => _isModelLoaded;

    void Start()
    {
        if (autoLoadOnStart)
        {
            LoadModel();
        }
    }

    /// <summary>
    /// 加载ONNX模型
    /// </summary>
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

            // 配置会话选项
            var sessionOptions = new SessionOptions();

            if (useGpu)
            {
                try
                {
                    sessionOptions.AppendExecutionProvider_CUDA(0);
                    Debug.Log("使用GPU加速");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"GPU加速不可用，回退到CPU: {ex.Message}");
                }
            }

            sessionOptions.InterOpNumThreads = 1;
            sessionOptions.IntraOpNumThreads = inferenceThreads;
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            // 加载模型
            _session = new InferenceSession(modelPath, sessionOptions);

            // 获取输入输出名称
            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();

            _isModelLoaded = true;

            if (showDebugInfo)
            {
                Debug.Log($"模型加载成功: {modelFileName}");
                Debug.Log($"输入名称: {_inputName}");
                Debug.Log($"输出名称: {_outputName}");

                var inputMeta = _session.InputMetadata[_inputName];
                var outputMeta = _session.OutputMetadata[_outputName];

                Debug.Log($"输入维度: [{string.Join(", ", inputMeta.Dimensions)}]");
                Debug.Log($"输出维度: [{string.Join(", ", outputMeta.Dimensions)}]");
            }

            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"加载模型失败: {ex.Message}\n{ex.StackTrace}");
            return false;
        }
    }

    /// <summary>
    /// 执行推理
    /// </summary>
    /// <param name="inputData">输入数据</param>
    /// <returns>输出数据</returns>
    public float[] Infer(float[] inputData)
    {
        if (!_isModelLoaded)
        {
            Debug.LogError("模型未加载！");
            return null;
        }

        if (inputData == null || inputData.Length == 0)
        {
            Debug.LogError("输入数据为空！");
            return null;
        }

        try
        {
            var startTime = Time.realtimeSinceStartup;

            // 创建输入张量 [batch_size=1, channels=1, length]
            var inputDimensions = new[] { 1, 1, inputData.Length };
            var inputTensor = new DenseTensor<float>(inputData, inputDimensions);

            // 创建输入容器
            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };

            // 执行推理
            float[] output;
            using (var results = _session.Run(inputs))
            {
                output = results.First().AsEnumerable<float>().ToArray();
            }

            lastInferenceTime = (Time.realtimeSinceStartup - startTime) * 1000f;

            if (showDebugInfo)
            {
                Debug.Log($"推理完成: 输入长度={inputData.Length}, 输出长度={output.Length}, 耗时={lastInferenceTime:F2}ms");
            }

            return output;
        }
        catch (Exception ex)
        {
            Debug.LogError($"推理失败: {ex.Message}\n{ex.StackTrace}");
            return null;
        }
    }

    /// <summary>
    /// 异步推理（协程）
    /// </summary>
    public IEnumerator InferAsync(float[] inputData, Action<float[]> onComplete)
    {
        if (!_isModelLoaded)
        {
            Debug.LogError("模型未加载！");
            onComplete?.Invoke(null);
            yield break;
        }

        float[] result = null;
        bool completed = false;

        // 在后台线程执行推理
        System.Threading.Tasks.Task.Run(() =>
        {
            try
            {
                result = Infer(inputData);
            }
            catch (Exception ex)
            {
                Debug.LogError($"异步推理失败: {ex.Message}");
            }
            finally
            {
                completed = true;
            }
        });

        // 等待完成
        while (!completed)
        {
            yield return null;
        }

        onComplete?.Invoke(result);
    }

    /// <summary>
    /// 从AudioClip提取数据并推理
    /// </summary>
    public float[] InferFromAudioClip(AudioClip audioClip)
    {
        if (audioClip == null)
        {
            Debug.LogError("AudioClip为空！");
            return null;
        }

        // 提取音频数据
        float[] samples = new float[audioClip.samples * audioClip.channels];
        audioClip.GetData(samples, 0);

        // 如果是立体声，转换为单声道
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

    /// <summary>
    /// 批量推理多个片段
    /// </summary>
    public List<float[]> InferBatch(List<float[]> inputBatch)
    {
        if (!_isModelLoaded)
        {
            Debug.LogError("模型未加载！");
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

    /// <summary>
    /// 性能基准测试
    /// </summary>
    public IEnumerator BenchmarkPerformance(int iterations = 100, int inputLength = 100)
    {
        if (!_isModelLoaded)
        {
            Debug.LogError("模型未加载！");
            yield break;
        }

        Debug.Log($"开始性能测试: {iterations}次推理, 输入长度={inputLength}");

        // 创建测试数据
        float[] testInput = new float[inputLength];
        System.Random random = new System.Random();
        for (int i = 0; i < inputLength; i++)
        {
            testInput[i] = (float)(random.NextDouble() * 2 - 1);
        }

        // 预热
        for (int i = 0; i < 10; i++)
        {
            Infer(testInput);
        }

        yield return new WaitForSeconds(0.5f);

        // 性能测试
        List<float> times = new List<float>();

        for (int i = 0; i < iterations; i++)
        {
            float startTime = Time.realtimeSinceStartup;
            Infer(testInput);
            float elapsedTime = (Time.realtimeSinceStartup - startTime) * 1000f;
            times.Add(elapsedTime);

            if (i % 10 == 0)
            {
                yield return null; // 每10次推理让出一帧
            }
        }

        // 统计结果
        float avgTime = times.Average();
        float minTime = times.Min();
        float maxTime = times.Max();

        Debug.Log($"=== 性能测试结果 ===");
        Debug.Log($"迭代次数: {iterations}");
        Debug.Log($"平均时间: {avgTime:F2} ms");
        Debug.Log($"最小时间: {minTime:F2} ms");
        Debug.Log($"最大时间: {maxTime:F2} ms");
        Debug.Log($"吞吐量: {1000.0f / avgTime:F2} inferences/sec");
    }

    void OnDestroy()
    {
        // 清理资源
        if (_session != null)
        {
            _session.Dispose();
            _session = null;
        }
        _isModelLoaded = false;
    }

    void OnApplicationQuit()
    {
        OnDestroy();
    }
}