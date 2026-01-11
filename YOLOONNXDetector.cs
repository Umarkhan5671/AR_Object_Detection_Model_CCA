using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;

public class YOLOONNXDetector : MonoBehaviour
{
    public NNModel onnxModelAsset;
    public float confidenceThreshold = 0.3f;
    public string[] classNames;

    private IWorker worker;
    private const int INPUT_SIZE = 640;

    void Start()
    {
        var model = ModelLoader.Load(onnxModelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
        Debug.Log("✅ ONNX Model Loaded");

        if (classNames == null || classNames.Length == 0)
        {
            classNames = new string[]
            {
                "person","bicycle","car","motorcycle","airplane","bus","train","truck"
            };
        }
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }

    public List<Detection> RunInference(Texture2D source)
    {
        List<Detection> detections = new List<Detection>();

        Texture2D resized = ResizeTexture(source, INPUT_SIZE, INPUT_SIZE);
        Tensor input = TextureToTensor(resized);

        worker.Execute(input);
        Tensor output = worker.PeekOutput();

        Debug.Log("📦 Output Shape: " + output.shape);

        ParseYOLO(output, detections);

        input.Dispose();
        output.Dispose();

        return detections;
    }

    // ================= YOLO PARSER =================
    void ParseYOLO(Tensor output, List<Detection> detections)
    {
        int rows = output.shape[0];   // N detections
        int cols = output.shape[1];   // 85 values

        for (int i = 0; i < rows; i++)
        {
            float objConf = output[i, 4];
            if (objConf < confidenceThreshold) continue;

            int classId = 0;
            float maxClass = 0f;

            for (int c = 0; c < classNames.Length; c++)
            {
                float classScore = output[i, 5 + c];
                if (classScore > maxClass)
                {
                    maxClass = classScore;
                    classId = c;
                }
            }

            float confidence = objConf * maxClass;
            if (confidence < confidenceThreshold) continue;

            float cx = output[i, 0] * INPUT_SIZE;
            float cy = output[i, 1] * INPUT_SIZE;
            float w  = output[i, 2] * INPUT_SIZE;
            float h  = output[i, 3] * INPUT_SIZE;

            Rect box = new Rect(cx - w / 2, cy - h / 2, w, h);

            detections.Add(new Detection
            {
                box = box,
                classId = classId,
                confidence = confidence
            });

            Debug.Log($"🎯 DETECTED: {classNames[classId]} {confidence:0.00}");
        }
    }

    // ================= HELPERS =================
    Texture2D ResizeTexture(Texture2D src, int w, int h)
    {
        RenderTexture rt = RenderTexture.GetTemporary(w, h);
        Graphics.Blit(src, rt);
        RenderTexture.active = rt;

        Texture2D tex = new Texture2D(w, h, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        tex.Apply();

        RenderTexture.ReleaseTemporary(rt);
        return tex;
    }

    Tensor TextureToTensor(Texture2D tex)
    {
        Color32[] pixels = tex.GetPixels32();
        Tensor t = new Tensor(1, INPUT_SIZE, INPUT_SIZE, 3);

        int idx = 0;
        for (int y = 0; y < INPUT_SIZE; y++)
        {
            for (int x = 0; x < INPUT_SIZE; x++)
            {
                Color32 c = pixels[idx++];
                t[0, y, x, 0] = c.r / 255f;
                t[0, y, x, 1] = c.g / 255f;
                t[0, y, x, 2] = c.b / 255f;
            }
        }
        return t;
    }

    public class Detection
    {
        public Rect box;
        public float confidence;
        public int classId;
    }
}
