// src/Frames.js
import React, { useEffect, useRef, useState, useCallback } from 'react';

const BACKEND_URL = 'http://localhost:5000/frames';
const FRAME_SEND_INTERVAL_MS = 33; // ~4 FPS. Adjust based on performance and backend needs.

const Frames = ({ jitsiVideoStream, onGlossDetected, onStatusUpdate, onTranslationReceived }) => {
  const videoRef = useRef(null);      // For rendering the Jitsi stream to capture from
  const canvasRef = useRef(null);     // Hidden canvas for drawing frames
  const [isSending, setIsSending] = useState(false);
  const [intervalId, setIntervalId] = useState(null);

  // Effect to handle the Jitsi video stream prop
  useEffect(() => {
    if (jitsiVideoStream && videoRef.current) {
      videoRef.current.srcObject = jitsiVideoStream;
      videoRef.current.play().catch(e => {
        console.error("Error playing Jitsi stream in hidden video element:", e);
        onStatusUpdate("Error playing Jitsi stream. Check console.");
      });
      onStatusUpdate('Jitsi stream connected. Ready to start gesture recognition.');
      // If isSending was true and stream became available, restart
      if(isSending && !intervalId) {
        startSendingFramesInternal();
      }
    } else {
      onStatusUpdate(isSending ? 'Jitsi stream lost. Stopping recognition.' : 'Waiting for Jitsi video stream...');
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject = null;
      }
      stopSendingFramesInternal(); // Stop if stream is lost or not available
    }

    return () => {
      // Don't stop tracks of jitsiVideoStream here; Jitsi owns its lifecycle.
      // Just clear our reference.
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject = null;
      }
    };
  }, [jitsiVideoStream, isSending]); // Added isSending to deps for potential auto-restart logic


  const captureAndSendFrame = useCallback(async () => {
    if (!videoRef.current || !videoRef.current.srcObject || videoRef.current.readyState < videoRef.current.HAVE_METADATA || !canvasRef.current) {
      console.warn("Video element not ready or stream not set for capturing.");
      return;
    }

    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;
    const ctx = canvasElement.getContext("2d", { willReadFrequently: true }); // Performance hint

    if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
      console.warn("Video dimensions are zero. Cannot capture frame.");
      return;
    }

    // Ensure canvas dimensions match the video for accurate capture
    if (canvasElement.width !== videoElement.videoWidth) canvasElement.width = videoElement.videoWidth;
    if (canvasElement.height !== videoElement.videoHeight) canvasElement.height = videoElement.videoHeight;

    // Draw current video frame from the Jitsi stream to the hidden canvas
    // Mirroring: Jitsi usually mirrors local video. If your model expects a non-mirrored
    // view (like looking in a mirror), you might not need to flip.
    // If Jitsi provides a non-mirrored stream and your model expects mirrored, or vice-versa, adjust here.
    // For now, direct draw:
    ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    canvasElement.toBlob(async (blob) => {
      if (!blob) {
        console.error("Failed to create blob from canvas.");
        onStatusUpdate("Error: Could not capture frame for sending.");
        return;
      }
      const formData = new FormData();
      formData.append("frame", blob, "frame.png");

      try {
        const response = await fetch(BACKEND_URL, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: "Unknown server error." }));
          onStatusUpdate(`Server Error: ${response.status} - ${errorData.error}`);
          console.error('Server error:', response.status, errorData);
          return;
        }

        const data = await response.json();
        onStatusUpdate(`Backend: ${data.status || 'No status from backend.'}`);

        if (data.predictions && data.predictions.length > 0) {
          onGlossDetected(data.predictions[0].gloss); // Send top gloss to App.js
          console.log("Prediction:", data.predictions[0].gloss);
        } else if (data.error) {
          console.error("Backend processing error:", data.error);
          onGlossDetected(''); // Clear gloss on error
        } else if (data.status && data.status.startsWith("Collecting")) {
          // Optionally, you could clear the gloss or show "..."
          // onGlossDetected("...");
        }
        if (data.translation) {
            onTranslationReceived(data.translation);
        }

      } catch (error) {
        onStatusUpdate(`Network Error: ${error.message}. Is backend running?`);
        console.error("Error sending frame:", error);
        // setIsSending(false); // Optional: stop sending on persistent network errors
      }
    }, "image/png");
  }, [onGlossDetected, onStatusUpdate, onTranslationReceived]); // Dependencies for useCallback

  const startSendingFramesInternal = useCallback(() => {
    if (intervalId) clearInterval(intervalId);
    if (jitsiVideoStream) {
      const id = setInterval(captureAndSendFrame, FRAME_SEND_INTERVAL_MS);
      setIntervalId(id);
      onStatusUpdate("Gesture recognition active. Sending frames...");
    } else {
      onStatusUpdate("Cannot start: Jitsi video stream not available.");
    }
  }, [intervalId, jitsiVideoStream, captureAndSendFrame, onStatusUpdate]);

  const stopSendingFramesInternal = useCallback(() => {
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
  }, [intervalId]);

  const toggleSending = () => {
    setIsSending((prevIsSending) => {
      const newIsSending = !prevIsSending;
      if (newIsSending) {
        if (jitsiVideoStream) {
          startSendingFramesInternal();
        } else {
          onStatusUpdate("Cannot start: Jitsi stream unavailable. Join with video.");
          return false; // Prevent toggling to true if no stream
        }
      } else {
        stopSendingFramesInternal();
        onStatusUpdate("Gesture recognition stopped by user.");
        onGlossDetected(""); // Clear gloss when stopping
      }
      return newIsSending;
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginTop: "20px" }}>
      {/* Hidden video element to render the Jitsi stream. Muted to prevent echo. */}
      <video ref={videoRef} style={{ display: "none" }} autoPlay playsInline muted />
      {/* Hidden canvas for frame grabbing */}
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <button
        onClick={toggleSending}
        style={{
          padding: "10px 20px",
          fontSize: "16px",
          fontWeight: "bold",
          color: "#fff",
          backgroundColor: isSending ? "#d9534f" : (jitsiVideoStream ? "#5cb85c" : "#cccccc"),
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
          transition: "background-color 0.3s ease",
          minWidth: "250px",
          marginBottom: "10px"
        }}
        disabled={!jitsiVideoStream && !isSending}
      >
        {isSending ? "Stop Gesture Recognition" : "Start Gesture Recognition"}
      </button>
    </div>
  );
};

export default Frames;