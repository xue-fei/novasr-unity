using System.Collections;
using System.IO;
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
        Loom.Initialize();

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
         
        if (testAudioClip != null)
        {
            float[] audioOutput = GetSamples(testAudioClip);
            audioOutput = modelInference.Infer(audioOutput);
            if (audioOutput != null)
            {
                Debug.Log($"示例3完成: 音频推理输出长度={audioOutput.Length}");
                SaveAudioAsWav(audioOutput, 48000, Application.dataPath + "/after.wav");
            }
        } 
    }

    float[] GetSamples(AudioClip audioClip)
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
        return samples;
    }

    /// <summary>
    /// 将 float[] 音频数据保存为 WAV 文件
    /// </summary>
    /// <param name="samples">单声道 float 音频数据（范围 -1.0 ~ 1.0）</param>
    /// <param name="sampleRate">采样率（如 22050, 44100）</param>
    /// <param name="filePath">保存路径（如 "output.wav"）</param>
    public void SaveAudioAsWav(float[] samples, int sampleRate, string filePath)
    {
        if (samples == null || samples.Length == 0)
        {
            Debug.LogError("音频数据为空！");
            return;
        }

        // 确保值在 [-1, 1] 范围内
        float[] clampedSamples = new float[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            clampedSamples[i] = Mathf.Clamp(samples[i], -1f, 1f);
        }

        // 转换为 16-bit PCM
        short[] pcm = new short[clampedSamples.Length];
        for (int i = 0; i < clampedSamples.Length; i++)
        {
            pcm[i] = (short)(clampedSamples[i] * 32767); // 32767 = Int16.MaxValue
        }

        // 写入 WAV 文件
        using (FileStream fs = new FileStream(filePath, FileMode.Create))
        using (BinaryWriter writer = new BinaryWriter(fs))
        {
            // RIFF header
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(36 + pcm.Length * 2); // ChunkSize
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));

            // fmt subchunk
            writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16); // Subchunk1Size
            writer.Write((short)1); // AudioFormat (1 = PCM)
            writer.Write((short)1); // NumChannels (1 = mono)
            writer.Write(sampleRate); // SampleRate
            writer.Write(sampleRate * 2); // ByteRate
            writer.Write((short)2); // BlockAlign
            writer.Write((short)16); // BitsPerSample

            // data subchunk
            writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            writer.Write(pcm.Length * 2); // Subchunk2Size
            foreach (short sample in pcm)
            {
                writer.Write(sample);
            }
        }

        Debug.Log($"✅ 音频已保存到: {filePath}");
    }
}