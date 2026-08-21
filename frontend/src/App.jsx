import React from 'react';
import StreamTester from './components/StreamTester';
import UploadTrigger from './components/UploadTrigger';
import RawDataViewer from './components/RawDataViewer';

export default function App() {
  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: '1000px', margin: '0 auto', padding: '20px' }}>
      <h1>NewsLens-AI — Phase 6.1 Functional Data Flow Test Client</h1>
      <p style={{ color: '#555' }}>
        Barebones unstyled test bench verifying end-to-end SSE streaming, file intake upload, and corpus/settings APIs.
      </p>
      <hr />

      <StreamTester />
      <UploadTrigger />
      <RawDataViewer />
    </div>
  );
}
