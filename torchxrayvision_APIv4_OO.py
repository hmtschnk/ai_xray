# Enhanced X-Ray Processing System - Speed Optimized
# Version: 4.0 Optimized

import os
import json
import shutil
import warnings
import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import multiprocessing as mp

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

import dicom2jpg
import torchxrayvision as xrv
import skimage.io
from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Configuration
STANDARD_OVERLAY_SIZE = (512, 512)
ORIGINAL_DISPLAY_IMG_KEY = "original_display_image"
PATHOLOGY_THRESHOLD = 0.3
ANATOMY_THRESHOLD = 0.3
WEIGHT = "densenet121-res224-all"

PATHOLOGY_CONFIDENCE = 0.4
ANATOMY_FLAG = 0

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def convert_dicom_to_jpg(dicom_path, output_dir, jpg_filename):
    """Convert DICOM file to JPG format"""
    try:
        dicom2jpg.dicom2jpg(dicom_path, output_dir)
        for file in os.listdir(output_dir):
            if file.lower().endswith(".jpg"):
                final_path = os.path.join(output_dir, jpg_filename)
                os.rename(os.path.join(output_dir, file), final_path)
                return final_path
        raise FileNotFoundError("JPG conversion failed - no JPG file found")
    except Exception as e:
        raise RuntimeError(f"Error converting DICOM to JPG: {e}")

class ImagePreprocessor:
    """Optimized image preprocessing with OpenCV acceleration"""
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _get_resize_params(input_shape, target_size):
        return target_size
    
    @staticmethod
    def load_and_preprocess_for_display(img_path):
        """Load and preprocess image for display with OpenCV optimization"""
        # Use OpenCV for faster loading
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = skimage.io.imread(img_path)
        
        # Handle color channels
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            elif img.shape[2] == 3 and img_path.lower().endswith(('.jpg', '.jpeg')):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize efficiently
        img_float = img.astype(np.float32)
        img_min, img_max = img_float.min(), img_float.max()
        img_normalized = (img_float - img_min) / (img_max - img_min + 1e-8)
        
        # Fast resize with OpenCV
        if img_normalized.shape[:2] != STANDARD_OVERLAY_SIZE:
            img_normalized = cv2.resize(img_normalized, STANDARD_OVERLAY_SIZE, 
                                      interpolation=cv2.INTER_LINEAR)
        
        return img_normalized
    
    @staticmethod
    def load_and_preprocess_for_model(img_path, target_size):
        """Load and preprocess image for model inference"""
        # Fast grayscale loading
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = skimage.io.imread(img_path)
            if len(img.shape) == 3:
                img = img.mean(2)
        
        # Normalize and resize
        img = img.astype(np.float32)
        img = xrv.datasets.normalize(img, 255)
        
        if img.shape != (target_size, target_size):
            img = cv2.resize(img, (target_size, target_size), 
                           interpolation=cv2.INTER_LINEAR)
        
        return img[None, ...]

class ModelManager:
    """Singleton for efficient model management"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_classification_model(self, device):
        """Get cached classification model"""
        if 'cls_model' not in self._models:
            model = xrv.models.DenseNet(weights=WEIGHT).to(device).eval()
            if device.type == 'cuda':
                model = model.half()
            self._models['cls_model'] = model
        return self._models['cls_model']
    
    def get_segmentation_model(self, device):
        """Get cached segmentation model"""
        if 'seg_model' not in self._models:
            model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
            if device.type == 'cuda':
                model = model.half()
            self._models['seg_model'] = model
        return self._models['seg_model']

class XRayProcessor:
    """High-performance X-ray processing with parallel execution"""
    
    def __init__(self, xray_file, output_path, unique_id, enable_mixed_precision=True):
        self.xray_file = xray_file
        self.output_path = output_path
        self.id = str(unique_id)
        self.enable_mixed_precision = enable_mixed_precision
        
        # Setup paths
        os.makedirs(self.output_path, exist_ok=True)
        self.jpg_path = os.path.join(output_path, f"{self.id}.jpg")
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model management
        self.model_manager = ModelManager()
        self.cls_model = self.model_manager.get_classification_model(self.device)
        self.seg_model = self.model_manager.get_segmentation_model(self.device)
        
        # Mixed precision setup
        self.use_amp = (self.device.type == 'cuda' and 
                       hasattr(torch.cuda, 'amp') and 
                       self.enable_mixed_precision)
        
        self._log(f"Initialized processor - Device: {self.device}, AMP: {self.use_amp}")
    
    def _log(self, message):
        """Centralized logging"""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}][PROCESSOR][{self.id}] {message}")
    
    def _prepare_image(self):
        """Prepare image file for processing"""
        self._log(f"Preparing image: {self.xray_file}")
        
        file_ext = os.path.splitext(self.xray_file)[-1].lower()
        
        if file_ext == ".dcm":
            return convert_dicom_to_jpg(self.xray_file, self.output_path, 
                                      os.path.basename(self.jpg_path))
        elif file_ext in [".jpg", ".png", ".jpeg"]:
            try:
                os.link(self.xray_file, self.jpg_path)
            except (OSError, NotImplementedError):
                shutil.copy2(self.xray_file, self.jpg_path)
            return self.jpg_path
        else:
            raise ValueError("Unsupported format. Only DICOM, JPG, and PNG supported.")
    
    @torch.no_grad()
    def _analyze_pathologies(self, img_path):
        """Analyze pathologies with mixed precision support"""
        self._log("Analyzing pathologies")
        
        img = ImagePreprocessor.load_and_preprocess_for_model(img_path, target_size=224)
        dtype = torch.float16 if self.use_amp else torch.float32
        input_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device, dtype=dtype)
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.cls_model(input_tensor)
        else:
            outputs = self.cls_model(input_tensor)
        
        results_array = outputs[0].cpu().float().numpy()
        results = dict(zip(self.cls_model.pathologies, map(float, results_array)))
        
        self._log("Pathology analysis complete")
        return results
    
    def _generate_disease_overlays(self, img_path, results):
        """Generate disease overlays only for pathologies above threshold"""
        self._log("Generating disease overlays")
        
        img = ImagePreprocessor.load_and_preprocess_for_model(img_path, target_size=224)
        dtype = torch.float16 if self.use_amp else torch.float32
        input_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device, dtype=dtype)
        
        # Get target layer
        target_layer = self.cls_model.features.denseblock3.denselayer16.conv2
        cam = GradCAM(model=self.cls_model, target_layers=[target_layer])
        
        overlays = {}
        
        # Only include pathologies above threshold
        significant_pathologies = [
            pathology for pathology, score in results.items()
            if score >= PATHOLOGY_CONFIDENCE
        ]
        
        if not significant_pathologies:
            self._log("No pathologies reached threshold. No overlays generated.")
            return overlays
        
        self._log(f"Pathologies above threshold: {significant_pathologies}")
        
        for idx, pathology in enumerate(self.cls_model.pathologies):
            if pathology not in significant_pathologies:
                continue  # Skip low-confidence findings
            
            target = ClassifierOutputTarget(idx)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    grayscale_cam = cam(input_tensor=input_tensor, targets=[target])[0]
            else:
                grayscale_cam = cam(input_tensor=input_tensor, targets=[target])[0]
            
            # Process heatmap
            cam_min, cam_max = grayscale_cam.min(), grayscale_cam.max()
            norm_heatmap = (grayscale_cam - cam_min) / (cam_max - cam_min + 1e-8)
            
            # Resize efficiently
            resized_heatmap = cv2.resize(norm_heatmap, STANDARD_OVERLAY_SIZE, 
                                         interpolation=cv2.INTER_LINEAR)
            
            # Apply threshold
            mask = resized_heatmap >= PATHOLOGY_THRESHOLD
            resized_heatmap[~mask] = np.nan
            
            overlays[pathology] = torch.from_numpy(resized_heatmap).float()
            self._log(f"Generated overlay for: {pathology}")
        
        self._log("Disease overlays complete")
        return overlays

    
    @torch.no_grad()
    def _generate_anatomical_overlays(self, img_path):
        """Generate anatomical overlays"""
        self._log("Generating anatomical overlays")
        
        img = ImagePreprocessor.load_and_preprocess_for_model(img_path, target_size=512)
        dtype = torch.float16 if self.use_amp else torch.float32
        input_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device, dtype=dtype)
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                pred = self.seg_model(input_tensor)
        else:
            pred = self.seg_model(input_tensor)
        
        pred_np = pred.cpu().float().squeeze(0).numpy()
        overlays = {}
        
        for i, region in enumerate(self.seg_model.targets):
            heatmap = pred_np[i]
            
            # Normalize
            hmap_min, hmap_max = heatmap.min(), heatmap.max()
            norm_heatmap = (heatmap - hmap_min) / (hmap_max - hmap_min + 1e-8)
            
            # Apply threshold
            mask = norm_heatmap >= ANATOMY_THRESHOLD
            active_area = norm_heatmap.copy()
            active_area[~mask] = np.nan
            
            # Resize if needed
            if active_area.shape != STANDARD_OVERLAY_SIZE:
                resized_area = cv2.resize(active_area, STANDARD_OVERLAY_SIZE, 
                                        interpolation=cv2.INTER_LINEAR)
            else:
                resized_area = active_area
            
            overlays[region] = torch.from_numpy(resized_area).float()
        
        self._log("Anatomical overlays complete")
        return overlays
    
    def _save_results(self, results):
        """Save analysis results to JSON"""
        json_path = os.path.join(self.output_path, f"{self.id}.json")
        output_data = {
            "id": self.id,
            "input_img": os.path.basename(self.xray_file),
            "results": results
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        
        self._log(f"Results saved to {json_path}")
    
    def _save_overlays(self, original_img, disease_overlays, anatomical_overlays):
        """Save all overlay data"""
        # Original overlay
        oo_path = os.path.join(self.output_path, f"OO-{self.id}.safetensors")
        save_file({ORIGINAL_DISPLAY_IMG_KEY: torch.from_numpy(original_img).float()}, oo_path)
        
        # Disease overlays
        disease_path = os.path.join(self.output_path, f"DO-{self.id}.safetensors")
        save_file(disease_overlays, disease_path)
        
        # Anatomical overlays (only if flag enabled)
        if ANATOMY_FLAG and anatomical_overlays:
            anatomy_path = os.path.join(self.output_path, f"AO-{self.id}.safetensors")
            save_file(anatomical_overlays, anatomy_path)
            self._log("Anatomical overlays saved")
        else:
            self._log("Skipping anatomical overlays (ANATOMY_FLAG=0)")
        
        self._log("All overlays saved")

    
    def process(self):
        """Main processing method with parallel execution"""
        try:
            self._log("Starting processing")
            
            # Step 1: Prepare image
            img_path = self._prepare_image()
            
            # Step 2: Parallel processing of independent tasks
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit parallel tasks
                display_future = executor.submit(
                    ImagePreprocessor.load_and_preprocess_for_display, img_path)
                pathology_future = executor.submit(self._analyze_pathologies, img_path)
                
                # Get results
                original_display_img = display_future.result()
                results = pathology_future.result()
            
            # Step 3: Save analysis results
            self._save_results(results)
            
            # Step 4: Generate overlays in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                disease_future = executor.submit(self._generate_disease_overlays, img_path, results)
                
                if ANATOMY_FLAG:
                    anatomy_future = executor.submit(self._generate_anatomical_overlays, img_path)
                else:
                    anatomy_future = None
                
                disease_overlays = disease_future.result()
                anatomical_overlays = anatomy_future.result() if anatomy_future else {}

            
            # Step 5: Save all overlays
            self._save_overlays(original_display_img, disease_overlays, anatomical_overlays)
            
            self._log("Processing completed successfully")
            return self.id, results
            
        except Exception as e:
            self._log(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None

class XRayVisualizer:
    """High-performance visualization with robust tensor handling"""
    
    def __init__(self, uuid_str, output_path):
        self.uuid = uuid_str
        self.output_path = output_path
        self.disease_results = None
        self.disease_data = None
        self.anatomical_data = None
        self.base_image_display = None
        self._load_data()
    
    def _log(self, message):
        """Centralized logging"""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}][VISUALIZER][{self.uuid}] {message}")
    
    def _load_json(self, json_path):
        """Load JSON data"""
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_data(self):
        """Load all visualization data with optional AO/DO/OO files"""
        self._log("Loading visualization data")
        
        try:
            json_path = os.path.join(self.output_path, f"{self.uuid}.json")
            disease_path = os.path.join(self.output_path, f"DO-{self.uuid}.safetensors")
            anatomy_path = os.path.join(self.output_path, f"AO-{self.uuid}.safetensors")
            oo_path = os.path.join(self.output_path, f"OO-{self.uuid}.safetensors")
            
            # Always required
            json_data = self._load_json(json_path)
            self.disease_results = json_data["results"]
            
            # Load optional safetensors
            self.disease_data = load_file(disease_path) if os.path.exists(disease_path) else {}
            #if not self.disease_data:
            #    self._log(f"Optional file missing: {disease_path}")
            
            self.anatomical_data = load_file(anatomy_path) if os.path.exists(anatomy_path) else {}
            #if not self.anatomical_data:
            #    self._log(f"Optional file missing: {anatomy_path}")
            
            oo_data = load_file(oo_path) if os.path.exists(oo_path) else {}
            if oo_data:
                self.base_image_display = oo_data[ORIGINAL_DISPLAY_IMG_KEY].cpu().numpy()
            else:
                self.base_image_display = None
                #self._log(f"Optional file missing: {oo_path}")
            
            # Ensure RGB format if image exists
            if self.base_image_display is not None:
                if len(self.base_image_display.shape) == 2:
                    self.base_image_display = np.stack([self.base_image_display] * 3, axis=-1)
                elif self.base_image_display.shape[-1] == 4:
                    self.base_image_display = self.base_image_display[..., :3]
            
            self._log("Data loaded successfully")
            
        except Exception as e:
            self._log(f"Data loading failed: {e}")
            raise

    
    def get_disease_results(self):
        """Get disease analysis results"""
        return self.disease_results
    
    def _safe_tensor_to_numpy(self, tensor):
        """Safely convert tensor to numpy array"""
        try:
            if torch.is_tensor(tensor):
                return tensor.detach().cpu().numpy()
            elif hasattr(tensor, 'cpu'):
                return tensor.cpu().numpy()
            elif hasattr(tensor, 'numpy'):
                return tensor.numpy()
            else:
                return np.array(tensor)
        except Exception as e:
            raise ValueError(f"Could not convert tensor to numpy: {e}")
    
    def show_overlays(self, keys, alphas=None, colormaps=None, return_bytes=False, dpi=100):
        """Generate overlay visualization with robust error handling"""
        if self.base_image_display is None:
            self._log("No base image available")
            return None
        
        self._log(f"Generating visualization for keys: {keys}")
        
        try:
            # Prepare base image
            composite_image = self.base_image_display.astype(np.float32, copy=True)
            if composite_image.max() > 1.5:
                composite_image /= 255.0
            
            # Ensure RGB format
            if composite_image.ndim == 2:
                composite_image = np.stack([composite_image] * 3, axis=-1)
            elif composite_image.ndim == 3:
                if composite_image.shape[-1] == 1:
                    composite_image = np.concatenate([composite_image] * 3, axis=-1)
                elif composite_image.shape[-1] == 4:
                    composite_image = composite_image[..., :3]
            
            # Handle NaN values
            np.nan_to_num(composite_image, copy=False)
            
            # Set defaults
            if alphas is None:
                alphas = [0.5] * len(keys)
            if colormaps is None:
                colormaps = [cm.jet] * len(keys)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
            ax.imshow(np.clip(composite_image, 0.0, 1.0))
            ax.axis('off')
            
            # Process each overlay
            for key, alpha, cmap in zip(keys, alphas, colormaps):
                self._process_single_overlay(key, alpha, cmap, composite_image, ax)
            
            # Final display
            ax.imshow(np.clip(composite_image, 0.0, 1.0))
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            if return_bytes:
                img_io = BytesIO()
                canvas = FigureCanvas(fig)
                canvas.print_png(img_io)
                plt.close(fig)
                img_io.seek(0)
                self._log("Visualization completed successfully")
                return img_io
            else:
                plt.show()
                return None
                
        except Exception as e:
            self._log(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_single_overlay(self, key, alpha, cmap, composite_image, ax):
        """Process a single overlay with robust error handling"""
        try:
            # Find overlay data
            overlay_tensor = None
            source = None
            
            if key in self.disease_data:
                overlay_tensor = self.disease_data[key]
                source = 'Disease'
            elif key in self.anatomical_data:
                overlay_tensor = self.anatomical_data[key]
                source = 'Anatomy'
            
            if overlay_tensor is None:
                self._log(f"Overlay not found for key: {key}")
                return
            
            # Convert to numpy safely
            overlay_data = self._safe_tensor_to_numpy(overlay_tensor)
            overlay_data = np.squeeze(overlay_data)
            
            if overlay_data.ndim != 2:
                self._log(f"Invalid overlay shape for {key}: {overlay_data.shape}")
                return
            
            # Check for valid data
            finite_mask = np.isfinite(overlay_data)
            finite_count = np.sum(finite_mask)
            
            if finite_count == 0:
                self._log(f"No finite values in overlay for {key}")
                return
            
            # Normalize data
            valid_data = overlay_data[finite_mask]
            minv, maxv = np.min(valid_data), np.max(valid_data)
            
            if maxv == minv:
                self._log(f"Constant values in overlay for {key}")
                return
            
            # Create normalized heatmap
            norm = np.full_like(overlay_data, np.nan, dtype=np.float32)
            norm[finite_mask] = (valid_data - minv) / (maxv - minv + 1e-8)
            
            # Apply colormap
            norm_clipped = np.clip(norm, 0.0, 1.0)
            heatmap_rgba = cmap(norm_clipped)
            heatmap_alpha = heatmap_rgba[..., 3] * float(alpha)
            
            # Create visibility mask
            alpha_threshold = 1e-3
            alpha_mask = heatmap_alpha > alpha_threshold
            visible_mask = finite_mask & alpha_mask
            visible_count = np.sum(visible_mask)
            
            if visible_count == 0:
                self._log(f"No visible pixels for {key}")
                return
            
            # Apply alpha blending
            alpha_values = heatmap_alpha[visible_mask]
            heatmap_rgb = heatmap_rgba[..., :3]
            
            for c in range(3):
                original = composite_image[visible_mask, c]
                overlay = heatmap_rgb[visible_mask, c]
                composite_image[visible_mask, c] = (
                    (1.0 - alpha_values) * original + alpha_values * overlay
                )
            
            # Add label
            self._add_overlay_label(key, source, norm, finite_mask, ax)
            
        except Exception as e:
            self._log(f"Failed to process overlay {key}: {e}")
    
    def _add_overlay_label(self, key, source, norm, finite_mask, ax):
        """Add label at maximum intensity point"""
        try:
            valid_norm = norm[finite_mask]
            if len(valid_norm) > 0:
                max_flat_idx = np.nanargmax(valid_norm)
                valid_coords = np.where(finite_mask)
                max_y = valid_coords[0][max_flat_idx]
                max_x = valid_coords[1][max_flat_idx]
                
                label_text = f"{source}: {key}"
                txt = ax.text(
                    int(max_x), int(max_y), label_text,
                    color='white', fontsize=12, weight='bold',
                    ha='center', va='center'
                )
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=1.5, foreground='black'),
                    path_effects.Normal()
                ])
        except Exception as e:
            self._log(f"Could not add label for {key}: {e}")

class BatchXRayProcessor:
    """Utility for batch processing multiple X-rays"""
    
    @staticmethod
    def process_batch(xray_files, output_path, max_workers=None):
        """Process multiple X-ray files in parallel"""
        if max_workers is None:
            max_workers = min(len(xray_files), mp.cpu_count())
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i, xray_file in enumerate(xray_files):
                processor = XRayProcessor(xray_file, output_path, f"batch_{i}")
                future = executor.submit(processor.process)
                futures.append(future)
            
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        
        return results