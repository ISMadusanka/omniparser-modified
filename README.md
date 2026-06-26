# OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent


## Install 
First clone the repo, and then install environment:
```python
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

Ensure you have the V2 weights downloaded in weights folder (ensure caption weights folder is called icon_caption_florence). If not download them with:
```
   # download the model checkpoints to local directory OmniParser/weights/
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
   mv weights/icon_caption weights/icon_caption_florence
```

<!-- ## [deprecated]
Then download the model ckpts files in: https://huggingface.co/microsoft/OmniParser, and put them under weights/, default folder structure is: weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2. 

For v1: 
convert the safetensor to .pt file. 
```python
python weights/convert_safetensor_to_pt.py

For v1.5: 
download 'model_v1_5.pt' from https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5, make a new dir: weights/icon_detect_v1_5, and put it inside the folder. No weight conversion is needed. 
``` -->

## Examples:
We put together a few simple examples in the demo.ipynb. 

## Gradio Demo
To run gradio demo, simply run:
```python
python gradio_demo.py
```

## FastAPI Server
An alternative to the Gradio demo is the FastAPI server, which exposes a REST API that can be called from tools like **Postman**, **cURL**, or any HTTP client.

### Running the Server
Install the additional dependencies (if not already installed):
```bash
pip install fastapi uvicorn python-multipart
```

Start the server:
```bash
python fastapi_server.py
```
The server will start at `http://localhost:8000`.

### API Endpoints

#### `GET /health`
Health check endpoint to verify the server is running.

**Response:**
```json
{
  "status": "healthy"
}
```

#### `POST /process`
Processes a screenshot image and returns parsed UI elements with bounding boxes.

**Request** (`multipart/form-data`):

| Parameter        | Type    | Required | Default | Description                                      |
|------------------|---------|----------|---------|--------------------------------------------------|
| `image`          | File    | Yes      | —       | The screenshot image to parse                    |
| `box_threshold`  | float   | No       | `0.05`  | Confidence threshold for bounding box detection   |
| `iou_threshold`  | float   | No       | `0.1`   | IoU threshold for removing overlapping boxes      |
| `use_paddleocr`  | bool    | No       | `true`  | Whether to use PaddleOCR for text recognition     |
| `imgsz`          | int     | No       | `640`   | Image size for icon detection                     |

**Response:**
```json
{
  "parsed_content_list": "icon 0: ...\nicon 1: ...",
  "label_coordinates": { ... },
  "image_base64": "<base64 encoded labeled image>"
}
```

### Using with Postman
1. Create a new **POST** request to `http://localhost:8000/process`.
2. Go to the **Body** tab and select **form-data**.
3. Add a key `image`, set its type to **File**, and select a screenshot image.
4. Optionally add `box_threshold`, `iou_threshold`, `use_paddleocr`, and `imgsz` as text fields.
5. Click **Send**.

### Using with cURL
```bash
curl -X POST http://localhost:8000/process \
  -F "image=@screenshot.png" \
  -F "box_threshold=0.05" \
  -F "iou_threshold=0.1" \
  -F "use_paddleocr=true" \
  -F "imgsz=640"
```

## Model Weights License
For the model checkpoints on huggingface model hub, please note that icon_detect model is under AGPL license since it is a license inherited from the original yolo model. And icon_caption_blip2 & icon_caption_florence is under MIT license. Please refer to the LICENSE file in the folder of each model: https://huggingface.co/microsoft/OmniParser.

## 📚 Citation
Our technical report can be found [here](https://arxiv.org/abs/2408.00203).
If you find our work useful, please consider citing our work:
```
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent}, 
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00203}, 
}
```
