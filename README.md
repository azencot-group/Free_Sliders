# Free_Sliders

Free_Sliders is a multimodal framework for generating concept sliders using text-to-image models. It supports multiple methods including FreeSliders and TE (Textual Embedding) approaches with Stable Diffusion as the backbone model.

**🎯 Multimodality Support:**
- ✅ **Text-to-Image**: Currently supported with Stable Diffusion
- 🚧 **Text-to-Video**: Code coming soon
- 🚧 **Text-to-Audio**: Code coming soon

The framework is designed to be extensible across different modalities, enabling concept sliders for various types of content generation.

## Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:azencot-group/Free_Sliders.git
   cd Free_Sliders
   ```

2. **Install dependencies:**
   Run the following in terminal:
   ```bash
   uv sync
   source ./venv/bin/activate
   ```

## Quick Start

### Option 1: Using the Interactive Jupyter Notebook (Recommended)

The easiest way to get started is using the interactive notebook:

```bash
jupyter notebook T2I_interactive.ipynb
```

This notebook provides:
- Step-by-step guidance through the pipeline
- Interactive parameter adjustment
- Visual output display
- Easy experimentation with different prompts and concepts

### Option 2: Command Line Interface

Run the main script with command line arguments:

```bash
python main.py --BaseModel T2I --method FreeSliders --scales -6 -3 0 4 8
```

#### Available Parameters:

**Required:**
- `--BaseModel`: Choose base model (`T2I`)

**Optional:**
- `--method`: Method to use (`FreeSliders` or `TE`, default: `FreeSliders`)
- `--config_file`: Configuration file path (default: `configs/config.yaml`)
- `--num_images`: Number of images to generate per concept (default: 10)
- `--prompt_index`: Use specific prompt index, -1 for all prompts (default: -1)
- `--neptune`: Enable Neptune logging (default: False)

**Method-specific parameters:**
- `--t`: T values for slider generation (default: 15)
- `--scales`: Scales for the model (multiple values allowed)
- `--guidance`: Guidance scale (default: 6.0)
- `--guidance2`: Secondary guidance scale (default: 1.0)

#### Example Commands:

**Basic usage with FreeSliders:**
```bash
python main.py --BaseModel T2I --method FreeSliders --scales -5 -2.5 0 3 6 9
```

**Generate images for a specific concept:**
```bash
python main.py --BaseModel T2I --method FreeSliders --prompt_index 0 --scales -5 -2.5 0 3 6 9
```

### Option 3: Manual Python Script

You can also integrate Free_Sliders into your own Python scripts:

```python
import importlib
from utils.config_utils import load_config_from_yaml
from utils.prompt_utils import load_prompts_from_yaml
from methods.FreeSliders import FreeSliders
from loggers import create_default_logger

# Load configuration
config = load_config_from_yaml("configs/config.yaml")
prompts = load_prompts_from_yaml("configs/image_prompts.yaml", [])

# Initialize base model
BaseModelClass = importlib.import_module("base_models.BaseModel_StableDiffusion").BaseModel
base_model = BaseModelClass(config, "manual_run")

# Encode prompts
text_embeddings = base_model.encode_prompts(prompts)

# Initialize method
method = FreeSliders(base_model)

# Create logger
logger = create_default_logger()

# Run the method
method(
    args=your_args_object,
    text_embeddings={"static": text_embeddings},
    prompts={"static": prompts},
    logger=logger
)
```

## Configuration

### Main Configuration (configs/config.yaml)

The main configuration file controls:
- **Model settings**: Pretrained model path, precision, dimensions
- **Network parameters**: LoRA rank and alpha values
- **Evaluation settings**: Batch size and type
- **Output paths**: Where to save results
- **Other options**: xformers usage, logging preferences

### Prompt Configuration (configs/image_prompts.yaml)

Define your concepts with:
- `neutral`: Base prompt
- `positive`: Positive direction prompt
- `negative`: Negative direction prompt
- `guidance_scale`: Guidance scale for this concept
- `batch_size`: Batch size for generation
- `tag`: Concept identifier

Example:
```yaml
- neutral: "A realistic image of a person."
  positive: "A realistic image of a person, smiling widely, very happy."
  negative: "A realistic image of a person, frowning, very sad."
  guidance_scale: 9
  batch_size: 1
  tag: "smiling"
```

## Output

Generated images are saved in the `outputs/` directory, organized by:
- Concept tag (e.g., `smiling`, `age`, `glasses`)
- Random seed subdirectories
- Individual image files

## Methods

### FreeSliders
A method for generating concept sliders without additional training.

### TE (Textual Embedding)
A textual embedding-based approach for concept manipulation.

## Logging

The framework supports multiple logging backends:
- **Print Logger**: Console output (default)
- **Neptune**: MLOps experiment tracking
- **Weights & Biases**: Experiment tracking
- **TensorBoard**: Visualization
- **MLflow**: ML lifecycle management

To use Neptune logging, create the required credential files in the `neptune/` directory.

## Troubleshooting

1. **CUDA out of memory**: Reduce `batch_size` in configuration or use fewer `num_images`
2. **Missing dependencies**: Install missing packages using pip
3. **Model download issues**: Ensure stable internet connection for Hugging Face model downloads
4. **Neptune errors**: Check credentials in `neptune/token.txt` and `neptune/project.txt`

## Examples

The `outputs/` directory contains example results for various concepts like:
- Age manipulation
- Facial expressions (smiling, surprised)
- Accessories (glasses, beard)
- Object conditions (damaged car, cluttered room)
- Food states (cooked food)
- Makeup (lipstick)

Run the code with different concepts to explore the capabilities of Free_Sliders!

## Future Roadmap

Free_Sliders is evolving into a comprehensive multimodal framework:

## Coming Soon
- 🎬 Video concept sliders
- 🎵 Audio concept sliders

Stay tuned for updates as we expand Free_Sliders to support the full spectrum of generative AI modalities!