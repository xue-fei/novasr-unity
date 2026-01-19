using System.Collections;
using UnityEngine;

/// <summary>
/// 使用示例MonoBehaviour
/// </summary>
public class OnnxInferenceExample : MonoBehaviour
{
    public OnnxModelInference modelInference;
    public AudioClip testAudioClip;

    void Start()
    {
        if (modelInference == null)
        {
            modelInference = GetComponent<OnnxModelInference>();
        }

        // 等待模型加载
        StartCoroutine(RunExamples());
    }

    IEnumerator RunExamples()
    {
        // 等待模型加载
        while (!modelInference.IsModelLoaded)
        {
            yield return new WaitForSeconds(0.1f);
        }

        Debug.Log("=== 开始示例 ===");

        // 示例1: 基本推理
        float[] testInput = new float[100];
        for (int i = 0; i < testInput.Length; i++)
        {
            testInput[i] = Mathf.Sin(i * 0.1f);
        }

        float[] output = modelInference.Infer(testInput);
        if (output != null)
        {
            Debug.Log($"示例1完成: 输出长度={output.Length}");
        }

        yield return new WaitForSeconds(0.5f);

        // 示例2: 异步推理
        yield return modelInference.InferAsync(testInput, (result) =>
        {
            if (result != null)
            {
                Debug.Log($"示例2完成: 异步推理输出长度={result.Length}");
            }
        });

        yield return new WaitForSeconds(0.5f);

        // 示例3: AudioClip推理
        if (testAudioClip != null)
        {
            float[] audioOutput = modelInference.InferFromAudioClip(testAudioClip);
            if (audioOutput != null)
            {
                Debug.Log($"示例3完成: 音频推理输出长度={audioOutput.Length}");
            }
        }

        yield return new WaitForSeconds(0.5f);

        // 示例4: 性能测试
        yield return modelInference.BenchmarkPerformance(100, 100);

        Debug.Log("=== 所有示例完成 ===");
    }
}