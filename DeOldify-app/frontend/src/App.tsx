import React, { useState, useRef, ChangeEvent, useEffect, CSSProperties } from 'react';
import { Upload } from 'lucide-react';
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';
import { ClimbingBoxLoader } from 'react-spinners';
import 'rsuite/dist/rsuite.min.css'
import { Slider } from 'rsuite';

const BACKEND_API_URL = import.meta.env.VITE_BACKEND_API_URL

interface ImageControls {
  brightness: number;
  contrast: number;
  sharpness: number;
}

const override: CSSProperties = {
  display: "block",
  margin: "0 auto",
  borderColor: "red",
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [upload, setUpload] = useState<Number | null>(null);
  const [processing, setProcessing] = useState<boolean | undefined>(false)
  const [finalFilename, setFinalFilename] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<string | null>(null)
  const [preview, setPreview] = useState<string | null>(null);
  const [processed, setProcessed] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [controls, setControls] = useState<ImageControls>({
    brightness: 50,
    contrast: 50,
    sharpness: 50,
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleUpload = async () => {
    const chunk_size = 1024 * 1024; // Chunk size set to 1 MB
    let offset = 0;
    let chunk_number = 0;
    let res: Response | null = null
    if (file) {
      // Loop until all chunks are uploaded
      while (offset < file?.size) {
        setUpload(0)
        // Slice the file into chunks
        const chunk = file.slice(offset, offset + chunk_size);

        // Create a blob from the chunk
        const chunk_blob = new Blob([chunk], { type: file.type });

        // Create a FormData object to send chunk data
        const formData = new FormData();
        formData.append("file", chunk_blob);
        formData.append("name", file.name);
        formData.append("chunk_number", String(chunk_number));
        formData.append(
          "total_chunks",
          String(Math.ceil(file?.size / chunk_size))
        );

        // Send the chunk data to the server using fetch API
        res = await fetch(`${BACKEND_API_URL}process/upload`, {
          method: "POST",
          body: formData,
        });

        setUpload(Math.ceil(file?.size / chunk_size))
        console.log()
        // Update offset and chunk number for the next iteration
        offset += chunk_size;
        chunk_number += 1;
      }
    }
    if (res){
      setFinalFilename((await res.json()).finalFilename)
    }
  }

  const handleFileProcessing = async () => {
    if (finalFilename) {
      setUpload(null)
      setProcessing(true)
      const params = new URLSearchParams()
      params.set("file_name", finalFilename)
      params.set("brightness", controls.brightness.toString())
      params.set("sharpness", controls.sharpness.toString())
      params.set("contrast", controls.contrast.toString())
      const url = BACKEND_API_URL + "process/process_data?" + params.toString()
      const res = await fetch(url)
      const return_url = URL.createObjectURL(await res.blob())
      setProcessed(return_url)
      setProcessing(false)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };
  // If file is submitted via dropping
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile)
    }
  };
  // If file is submitted via viewer
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile){
      setFile(selectedFile)
    }
  };

  const updateControl = (control: keyof ImageControls, value: number) => {
    setControls(prev => ({
      ...prev,
      [control]: value
    }));
  };

  const handleControlChange = (control: keyof ImageControls, value: string) => {
    updateControl(control, Number(value));
  };

  useEffect(() => {
    if (file !== null) {
      const url = URL.createObjectURL(file)
      setPreview(url)
      setMediaType(file.type[0])
      setProcessing(false)
      handleUpload()
    }
  }, [file]);

  useEffect(() => {
    if (finalFilename!== null) {
      handleFileProcessing()
    }
  }, [finalFilename, controls])

  console.log(preview, processed)

  return (
    <div className="min-h-screen bg-gray-300 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-800 mb-8">Noir-ify</h1>

        {/* Upload Section */}
        {!processing && upload === null && !processed && (<div
          className={`border-2 border-dashed rounded-lg p-8 mb-8 text-center transition-colors ${
            isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*,video/*"
            className="hidden"
          />
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 bg-blue-100 rounded-full">
              <Upload className="w-8 h-8 text-blue-600" />
            </div>
            <div>
              <p className="text-lg font-medium">Drag and drop your file here</p>
              <p className="text-gray-500">or</p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="mt-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Browse Files
              </button>
            </div>
            <p className="text-sm text-gray-500">Supports images and videos {'('}video file {'>'}50mb processing can take a lot of time{')'}</p>
          </div>
        </div>)}

        {/* Uploading bar */}
        {upload !== null && (
          <div>
            <h2>Upload In Progress</h2>
            <progress value={upload.toString()}/>
          </div>
        )}

        {/* Processing bar */}
        {processing && (
          <div>
            <h1>Noirifying your file...Please wait</h1>
            <ClimbingBoxLoader
              color={'#8FDFB6'}
              loading={processing}
              cssOverride={override}
              size={40}
              aria-label="Loading Spinner"
              data-testid="loader"
            />
          </div>
        )}

        {/* Preview Section */}
        {preview && processed && !processing && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div
              ref={containerRef}
              className="mb-6 overflow-hidden rounded-lg"
            >
              {mediaType === 'i' ? (
                <ReactCompareSlider
                  boundsPadding={0}
                  itemOne = { <ReactCompareSliderImage src={preview}/>}
                  itemTwo = {<ReactCompareSliderImage src={processed}/>}
                />
                ) : (
                <ReactCompareSlider
                    boundsPadding={0}
                    itemOne={
                        <video
                            ref={videoRef}
                            src={preview}
                            muted
                            autoPlay
                            loop
                            playsInline
                            controls
                            className="w-full h-auto object-contain"
                        >
                        </video>
                    }
                    itemTwo={
                        <video
                            ref={videoRef}
                            src={processed}
                            muted
                            autoPlay
                            loop
                            playsInline
                            controls
                            className="w-full h-auto object-contain"
                        >
                        </video>
                    }
                />
              )}
            </div>


            {/* Controls */}
            <div className="space-y-4">
              <h3>Brightness</h3>
              <Slider
                defaultValue={controls.brightness}
                progress
                onChangeCommitted={(value) => handleControlChange('brightness', value.toString())}
              />
              <h3>Sharpness</h3>
              <Slider
                defaultValue={controls.sharpness}
                progress
                onChangeCommitted={(value) => handleControlChange('sharpness', value.toString())}
              />
              <h3>Contrast</h3>
              <Slider
                defaultValue={controls.contrast}
                progress
                onChangeCommitted={(value) => handleControlChange('contrast', value.toString())}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;