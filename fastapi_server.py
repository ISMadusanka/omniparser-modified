from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import io
import base64
from PIL import Image
import torch

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

app = FastAPI(title="OmniParser API")

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"})

# Initialize models
print("Initializing models...")
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Models initialized.")

@app.post("/process")
async def process_image(
    image: UploadFile = File(...),
    box_threshold: float = Form(0.05),
    iou_threshold: float = Form(0.1),
    use_paddleocr: bool = Form(True),
    imgsz: int = Form(640)
):
    try:
        contents = await image.read()
        image_input = Image.open(io.BytesIO(contents)).convert("RGB")
        
        box_overlay_ratio = image_input.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_input, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_input, 
            yolo_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=caption_model_processor, 
            ocr_text=text,
            iou_threshold=iou_threshold, 
            imgsz=imgsz
        )
        
        parsed_content_str = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
        
        return JSONResponse(content={
            "parsed_content_list": parsed_content_str,
            "label_coordinates": label_coordinates,
            "image_base64": dino_labled_img # Base64 encoded string of the labeled image
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=False)
