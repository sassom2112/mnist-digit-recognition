// src/static/canvas.js
import React, { useRef, useState, useEffect } from 'react';
import '../../src/DigitRecognizer.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faEraser } from '@fortawesome/free-solid-svg-icons';

function Canvas({ onResult }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const debounceRef = useRef(null);

  const GRID_SIZE = 28;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const getPos = (e) => {
    const canvas = canvasRef.current;
    if (e.touches) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: e.touches[0].clientX - rect.left,
        y: e.touches[0].clientY - rect.top,
      };
    }
    return { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY };
  };

  const startDrawing = (e) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.lineCap = 'round';
    ctx.lineWidth = 20;
    ctx.strokeStyle = 'white';
    ctx.beginPath();
    const { x, y } = getPos(e);
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = (e) => {
    e.preventDefault();
    const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    ctx.closePath();
    setIsDrawing(false);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(sendToBackend, 150);
  };

  const captureGridPixels = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const cellSize = canvas.width / GRID_SIZE;
    const pixels = [];

    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        const imageData = ctx.getImageData(j * cellSize, i * cellSize, cellSize, cellSize);
        let total = 0;
        for (let k = 0; k < imageData.data.length; k += 4) {
          const grayscale = (imageData.data[k] + imageData.data[k + 1] + imageData.data[k + 2]) / 3;
          total += (grayscale / 255) * (imageData.data[k + 3] / 255);
        }
        pixels.push(total / (imageData.data.length / 4));
      }
    }
    return pixels;
  };

  const centerImage = (pixels) => {
    let image = [];
    for (let i = 0; i < GRID_SIZE; i++) {
      image.push(pixels.slice(i * GRID_SIZE, (i + 1) * GRID_SIZE));
    }

    let xmin = GRID_SIZE, xmax = -1, ymin = GRID_SIZE, ymax = -1;
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        if (image[y][x] > 0.1) {
          if (x < xmin) xmin = x;
          if (x > xmax) xmax = x;
          if (y < ymin) ymin = y;
          if (y > ymax) ymax = y;
        }
      }
    }

    if (xmin > xmax || ymin > ymax) return new Array(GRID_SIZE * GRID_SIZE).fill(0);

    let cropped = [];
    for (let y = ymin; y <= ymax; y++) {
      cropped.push(image[y].slice(xmin, xmax + 1));
    }

    let centered = new Array(GRID_SIZE).fill(0).map(() => new Array(GRID_SIZE).fill(0));
    const offsetX = Math.floor((GRID_SIZE - (xmax - xmin + 1)) / 2);
    const offsetY = Math.floor((GRID_SIZE - (ymax - ymin + 1)) / 2);

    for (let y = 0; y < cropped.length; y++) {
      for (let x = 0; x < cropped[0].length; x++) {
        centered[y + offsetY][x + offsetX] = cropped[y][x];
      }
    }
    return centered.flat();
  };

  const sendToBackend = () => {
    let pixels = captureGridPixels();
    pixels = centerImage(pixels);

    if (pixels.length !== 784 || pixels.every(p => p < 0.05)) return;

    fetch(process.env.REACT_APP_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pixels }),
    })
      .then(res => res.ok ? res.json() : Promise.reject(res))
      .then(data => onResult(data))
      .catch(err => console.error('Prediction error:', err));
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    onResult(null);
  };

  const submitManually = () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    sendToBackend();
  };

  return (
    <div className="action-card">
      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          className="digit-canvas"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
        <div className="overlay">Draw a number between 0 and 9</div>
      </div>
      <div className="button-group">
        <button onClick={submitManually} className="action-button submit-btn" aria-label="Submit Drawing">
          Submit
        </button>
        <button onClick={clearCanvas} className="action-button" aria-label="Clear Canvas">
          <FontAwesomeIcon icon={faEraser} /> Clear
        </button>
      </div>
    </div>
  );
}

export default Canvas;
