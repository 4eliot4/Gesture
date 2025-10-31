// src/App.js
import React, { useEffect, useState, useCallback, useRef } from 'react';
import { JitsiMeeting } from '@jitsi/react-sdk';
import Frames from './Frames';

export default function App() {
  const [jitsiVideoStream, setJitsiVideoStream] = useState(null);
  const jitsiApi = useRef(null);
  const [currentDetectedGloss, setCurrentDetectedGloss] = useState('');
  const [currentTranslation,    setCurrentTranslation   ] = useState('');
  const [recognitionStatus, setRecognitionStatus] = useState('Initializing...');

  // Acquire local camera once on mount
  useEffect(() => {
    setRecognitionStatus('Requesting local camera access...');
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        setJitsiVideoStream(stream);
        setRecognitionStatus('Local camera stream acquired.');
      })
      .catch(error => {
        console.error('getUserMedia error:', error);
        setRecognitionStatus(`Error accessing camera: ${error.message}`);
      });
  }, []);

  // Handle Jitsi External API ready
  const handleApiReady = useCallback((api) => {
    console.log('Jitsi API is ready', api);
    jitsiApi.current = api;
    setRecognitionStatus('Jitsi API connected.');

    api.addEventListener('videoConferenceJoined', () => {
      setRecognitionStatus('Joined Jitsi conference.');
    });

    api.addEventListener('videoConferenceLeft', () => {
      setRecognitionStatus('Left Jitsi conference.');
      setCurrentDetectedGloss('');
      api.executeCommand('displayName', 'Gesture User');
    });
  }, []);

  const handleGlossDetected = useCallback(gloss => {
    setCurrentDetectedGloss(gloss);
  }, []);
  const handleTranslationReceived = useCallback(sentence => {
        if (sentence && jitsiApi.current) {
          jitsiApi.current.executeCommand(
            'displayName',
            `Gesture User: ${sentence}`
          );
        }
      }, []);
     

  const handleStatusUpdate = useCallback(status => {
    setRecognitionStatus(status);
  }, []);

  const YOUR_JITSI_DOMAIN = 'meet.jit.si';
  const ROOM_NAME = 'Ad=_123rsA';

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: 'auto' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '20px', fontSize: '24px', fontWeight: 'bold' }}>
        Jitsi Meet with Live Sign Language Recognition
      </h1>
      <div style={{ border: '1px solid #ccc', marginBottom: '20px', minHeight: '500px' }}>
        <JitsiMeeting
          key={`${YOUR_JITSI_DOMAIN}-${ROOM_NAME}`}
          domain={YOUR_JITSI_DOMAIN}
          roomName={ROOM_NAME}
          configOverwrite={{
            startWithAudioMuted: true,
            startWithVideoMuted: false,
            disableModeratorIndicator: true,
            prejoinPageEnabled: false,
          }}
          interfaceConfigOverwrite={{
            DISABLE_JOIN_LEAVE_NOTIFICATIONS: true,
          }}
          userInfo={{ displayName: 'Gesture User' }}
          onApiReady={handleApiReady}
          getIFrameRef={iframe => {
            if (iframe) {
              iframe.style.width = '100%';
              iframe.style.height = '500px';

              // Silence the speaker-selection warning
              const allowAttr = iframe.getAttribute('allow') || '';
              const filtered = allowAttr
                .split(';')
                .map(s => s.trim())
                .filter(s => s !== 'speaker-selection')
                .join('; ');
              iframe.setAttribute('allow', filtered);
            }
          }}
        />
      </div>
      <div style={{ marginTop: '20px', padding: '15px', border: '1px solid #eee', borderRadius: '5px', backgroundColor: '#f9f9f9' }}>
        <h2 style={{ marginTop: 0, marginBottom: '10px', fontSize: '20px' }}>Gesture Recognition Control</h2>
        <p style={{ marginBottom: '10px', fontStyle: 'italic' }}>Status: {recognitionStatus}</p>
        <Frames
          jitsiVideoStream={jitsiVideoStream}
          onGlossDetected={handleGlossDetected}
          onStatusUpdate={handleStatusUpdate}
          onTranslationReceived={handleTranslationReceived}
        />
        {currentDetectedGloss && (
          <div style={{ marginTop: '15px', fontSize: '18px', fontWeight: 'bold', color: '#333' }}>
            Detected Gloss: <span style={{ color: '#28a745' }}>{currentDetectedGloss}</span>
          </div>
        )} 
        {currentTranslation && (
          <div style={{ marginTop: '15px', fontSize: '18px', fontWeight: 'bold', color: '#333' }}>
            Translation: <span style={{ color: '#007bff' }}>{currentTranslation}</span>
          </div>
        )}
      </div>
    </div>
  );
}
