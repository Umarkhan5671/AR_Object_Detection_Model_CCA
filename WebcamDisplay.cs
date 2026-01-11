using UnityEngine;

public class WebcamDisplay : MonoBehaviour
{
    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        if (devices.Length == 0)
        {
            Debug.LogError("No camera detected");
            return;
        }

        WebCamTexture camTexture = new WebCamTexture(devices[0].name);
        Renderer renderer = GetComponent<Renderer>();
        renderer.material.mainTexture = camTexture;
        camTexture.Play();

        Debug.Log("Using webcam: " + devices[0].name);
    }
}
