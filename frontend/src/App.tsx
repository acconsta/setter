// From: https://github.com/tensorflow/tfjs-models/blob/master/coco-ssd/src/index.ts
import React from 'react';
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import './App.css';

type BBox = number[]
const NUM_CLASSES = 14
interface DetectionObject {
  label: string
  score: number
  bbox: BBox
}
const THRESHOLD = 0.85;
// Should exactly match ordre in util.py
const LABELS = {
  color: ["🟥", "🟩", "🟪"],
  shape: ["diamond", "oval", "squiggle"],
  number: ["1", "2", "3"],
  texture: ["solid", "striped", "open"],
}
const LABEL_VALUES = Object.values(LABELS)

function getLabelString(classes: number[]) {
  console.assert(typeof (classes[0]) === "number")
  console.assert(classes.length = NUM_CLASSES)
  let label_string: string[] = []
  for (let i = 0; i < 4; ++i) {
    let max_j = 0
    const start = 3 * i + 2
    const end = start + 3
    for (let j = start; j < end; j++) {
      if (classes[j] > classes[max_j]) {
        max_j = j
      }
    }
    label_string.push(LABEL_VALUES[i][max_j - start])
  }
  return label_string.join(" ")
}

function clip(height: number, width: number, aspectRatio: number) {
  let r = width / height
  if (r > aspectRatio) {
    // Too wide
    width = aspectRatio * height
  } else {
    height = width / aspectRatio
  }
  return [height, width]
}

async function load_model() {
  const model = await loadGraphModel("web_model/model.json");
  return model;
}

class App extends React.Component {
  videoRef = React.createRef<HTMLVideoElement>();
  canvasRef = React.createRef<HTMLCanvasElement>();

  componentDidMount() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise: Promise<void> = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "environment"
          }
        })
        .then(stream => {
          if (!this.videoRef.current) return
          this.videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            if (!this.videoRef.current) return
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
              if (!this.videoRef.current) return
              if (!this.canvasRef.current) return
              let v = this.videoRef.current
              const aspectRatio = v.videoWidth / v.videoHeight
              // Resize the video and canvas now that we know the aspect ratio
              let width: number = window.innerWidth
              let height: number = window.innerHeight
              let [height_clip, width_clip] = clip(height, width, aspectRatio)
              v.width = width_clip
              v.height = height_clip
              this.canvasRef.current.width = width_clip
              this.canvasRef.current.height = height_clip
            };
          });
        });

      const modelPromise = load_model();

      Promise.all([modelPromise, webCamPromise])
        .then(values => {
          if (!this.videoRef.current) return
          this.detectFrame(this.videoRef.current, values[0]);
        })
        .catch(error => {
          console.error(error);
        });
    }
  }

  detectFrame = (video: HTMLVideoElement, model: tf.GraphModel) => {
    tf.engine().startScope();
    model.executeAsync(this.process_input(video)).then(predictions => {
      if (!Array.isArray(predictions)) return
      this.renderPredictions(predictions);
      requestAnimationFrame(() => {
        // setTimeout(() => {
        this.detectFrame(video, model);
      })
      // }, 5000);
      tf.engine().endScope();
    });
  };

  process_input(video_frame: HTMLVideoElement) {
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    const expandedimg = tfimg.transpose([0, 1, 2]).expandDims();
    return expandedimg;
  };

  buildDetectedObjects(boxes: number[][][], classes_list: number[][][]) {
    // console.log(boxes, classes_list)
    const detectionObjects: DetectionObject[] = []
    var video_frame = document.getElementById('frame');

    classes_list[0].forEach((classes: number[], i: number) => {
      if (!video_frame) return
      const score = classes[1]
      if (score > THRESHOLD) {
        const bbox: BBox = [];
        const minY = boxes[0][i][0] * video_frame.offsetHeight;
        const minX = boxes[0][i][1] * video_frame.offsetWidth;
        const maxY = boxes[0][i][2] * video_frame.offsetHeight;
        const maxX = boxes[0][i][3] * video_frame.offsetWidth;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        detectionObjects.push({
          label: getLabelString(classes),  // TODO
          score: score,
          bbox: bbox
        })
      }
    })
    return detectionObjects
  }

  renderPredictions(predictions: tf.Tensor[]) {
    if (!this.canvasRef.current) return
    const ctx = this.canvasRef.current.getContext("2d");
    if (!ctx) return
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    //Getting predictions
    const boxes: any = predictions[4].arraySync();
    const classes: any = predictions[3].arraySync();
    const detections = this.buildDetectedObjects(boxes, classes);

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];
      const width = item['bbox'][2];
      const height = item['bbox'][3];

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(0) + "%").width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(item["label"] + " " + (100 * item["score"]).toFixed(0) + "%", x, y);
    });
  };

  render() {
    return (
      <div>
        {/* TODO: styling and mobile layout */}
        <video
          // style={{ height: height, width: width }}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          // width={width}
          // height={height}
          id="frame"
        />
        <canvas
          className="size"
          ref={this.canvasRef}
        // width={width}
        // height={height}
        />
      </div>
    );
  }
}

export default App;
