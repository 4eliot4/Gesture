import React, { useState, useEffect } from 'react';

const Playground = () => {
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    fetch('http://127.0.0.1:5000/time', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })
    .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        if (data && typeof data.time === 'number') {
          setCurrentTime(data.time);
        } else {
          console.error('Invalid JSON structure or data type:', data);
        }
      })
      .catch(error => {
        console.error('Error fetching time:', error.message || error);
      });
  }, []);

  return (
    <div >
      <header>

        ... no changes in this part ...

        <p>The current time is {currentTime}.</p>
      </header>
    </div>
  );
}

export default Playground