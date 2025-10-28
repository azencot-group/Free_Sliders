import imageio
import neptune.types
from PIL import Image, ImageDraw, ImageFont
import os
import gc
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import neptune
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import random
import torch
from typing import List, Tuple

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)

def plot_latents(mapped_latents, prompts, base_model, logger=None, tag="composite", t=None, experiment="Misc",
                 metric_scores_df=None):

    with torch.no_grad():
        all_videos = []
        column_headers = list(mapped_latents.keys())  # Extract column headers
        for latents in mapped_latents.values():
            frames = base_model.decode_latents(latents)
            frames = frames.squeeze(0)
            frames = (frames * 255).round().astype("uint8")
            # Readjust size for neptune
            resize_size = (frames.shape[2] // 2, frames.shape[1] // 2) # width, height
            pil_frames = [Image.fromarray(frame).convert("RGB").resize(resize_size, Image.LANCZOS) for frame in
                          frames]
            all_videos.append(pil_frames)

        # Assume all videos have the same frame dimensions.
        frame_width = all_videos[0][0].width
        frame_height = all_videos[0][0].height
        composite_width = len(all_videos) * frame_width

        # Set header heights
        overall_header_height = 200  # Header for prompts
        column_header_height = 40  # Header for each individual GIF
        table_height = 500 if metric_scores_df is not None else 0  # Adjust based on table presence
        total_header_height = overall_header_height + column_header_height

        # Create the header text (each label on its own line).
        overall_header_text = "\n".join(prompts)
        if t is not None:
            overall_header_text = overall_header_text + f"\nt = {t}"

        try:
            font_large = ImageFont.truetype("DejaVuSans.ttf", size=40)
            font_small = ImageFont.truetype("DejaVuSans.ttf", size=30)
        except Exception:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        composite_frames = []
        # For each frame index, build a composite frame.
        for frame_idx in range(base_model.num_frames):
            # Create an image for the row (concatenated frames).
            row_img = Image.new("RGB", (composite_width, frame_height))
            for col_idx, video in enumerate(all_videos):
                row_img.paste(video[frame_idx], (col_idx * frame_width, 0))
            # Adjust total height to fit prompts, GIFs, and table
            table_height = (len(metric_scores_df.columns) + 1) * 60 if metric_scores_df is not None else 0  # Table height based on rows
            final_height = total_header_height + frame_height + table_height + 10  # Add spacing
            # Create a new image to hold everything
            final_frame = Image.new("RGB", (composite_width, final_height), "white")
            # Draw on the new image
            draw = ImageDraw.Draw(final_frame)

            # **1. Draw the prompts (top)**
            draw.multiline_text((10, 10), overall_header_text, fill="black", font=font_large, align="left")

            # **2. Center the GIF row between the prompts and table**
            gif_y_start = total_header_height  # Position after prompts
            final_frame.paste(row_img, (0, gif_y_start))
            # Draw headers for each column (above each GIF)
            for col_idx, column_header in enumerate(column_headers):
                column_header = str(column_header)
                text_x = col_idx * frame_width + frame_width // 2
                text_y = total_header_height - 30  # Below the overall header
                text_width = draw.textlength(column_header, font=font_small)
                draw.text((text_x - text_width // 2, text_y), column_header, fill="black", font=font_small)

            # **3. Draw table at the bottom**
            if metric_scores_df is not None:
                table_start_y = gif_y_start + frame_height + 10  # Below GIFs
                metric_name_padding = 150
                cell_width = composite_width // (len(metric_scores_df.index) + 2)
                cell_height = 50
                flipped_df = metric_scores_df.T  # Transpose the DataFrame

                # Draw row headers (metrics on the left)
                for row_idx, metric in enumerate(flipped_df.index):
                    y = table_start_y + (row_idx + 1) * cell_height
                    draw.text((10, y), str(metric), fill="black", font=font_small)
                # Draw column headers (scales above each column)
                for col_idx, scale in enumerate(flipped_df.columns):
                    x = (col_idx + 1) * cell_width + metric_name_padding
                    draw.text((x, table_start_y), str(scale), fill="black", font=font_small)
                # Draw table values
                for row_idx, (index, row) in enumerate(flipped_df.iterrows()):
                    y = table_start_y + (row_idx + 1) * cell_height
                    for col_idx, value in enumerate(row):
                        x = (col_idx + 1) * cell_width + metric_name_padding
                        draw.text((x, y), f"{value:.4f}", fill="black", font=font_small)

                # Define the x-position for the vertical separator (right before the overall column)
                separator_x = len(flipped_df.columns) * cell_width + metric_name_padding - 10  # Adjust as needed

                # Draw the vertical line separating the overall column
                draw.line(
                    [(separator_x, table_start_y),
                     (separator_x, table_start_y + (len(flipped_df.index) + 1) * cell_height)],
                    fill="black",
                    width=2
                )

                # Draw the vertical line separating the row headers
                draw.line(
                    [(cell_width + metric_name_padding - 10, table_start_y),
                     (cell_width + metric_name_padding - 10, table_start_y + (len(flipped_df.index) + 1) * cell_height)],
                    fill="black",
                    width=2
                )

                # Draw the horizontal line separating headers from values
                header_line_y = table_start_y + cell_height - 5  # Right after the column headers
                draw.line(
                    [(10, header_line_y), (x + 100, header_line_y)],  # Extend slightly past the last column for balance
                    fill="black",
                    width=2
                )

            # Save final frame
            composite_frames.append(final_frame)

        # Save the composite frames as a single GIF.
        composite_gif_path = f"generated_videos/plotted_{prompts[0]}_t{t}.gif"
        os.makedirs(os.path.dirname(composite_gif_path), exist_ok=True)
        imageio.mimsave(composite_gif_path, composite_frames, format="GIF", duration=100, loop=0)

        # Optionally, upload the composite GIF to Neptune.
        if logger is not None:
            logger[f"{experiment}/gifs/{tag}"].log(neptune.types.File(composite_gif_path))

        del (
             composite_frames,
             all_videos,
             pil_frames,
             frames,
             metric_scores_df,
             # flipped_df,
             mapped_latents,
             column_headers,

             )
        flush()

import pickle

def image_loader(image):
    # desired size of the output image
    imsize = 64
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    image = (image-0.5)*2
    return image # image.to(torch.float)


def image_sliders(scaled_imgs, method, run_id=1):

    image_list = list(scaled_imgs.values())
    scales = list(scaled_imgs.keys())
    fig, ax = plt.subplots(1, len(image_list), figsize=(len(image_list) * 2, 2))
    plt.subplots_adjust(wspace=0.05, hspace=0)

    # Ensure ax is always iterable
    if len(image_list) == 1:
        ax = [ax]

    for i, a in enumerate(ax):
        a.imshow(image_list[i])
        a.set_title(f"{scales[i]}", fontsize=15)
        a.axis('off')

    # plt.tight_layout()
    # plt.show()
    plt.close()

    os.makedirs("outputs/sd", exist_ok=True)
    fig_path = f"outputs/sd/{method}_slider_{run_id}.png"
    fig.savefig(fig_path)

    return fig_path

def plot_scores(scores1, label1, title, run_id=1):

    os.makedirs("outputs/sd", exist_ok=True)
    output_path = f"outputs/sd/{title}_plot_{run_id}.png"
    scales = list(scores1.keys())
    y1 = list(scores1.values())
    # y1 = [scores1[scale] for scale in scales]

    # Create the plot
    plt.figure(figsize=(6, 4))
    plt.plot(scales, y1, marker='o', color="blue", label=label1)

    plt.xlabel('Slider Scale', fontsize=12, weight='bold')
    plt.ylabel(title, fontsize=12, weight='bold')
    plt.xticks(scales)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    # plt.show()
    plt.close()

    return output_path


def display_one_slider_and_overall_scores(images, tag, logger, scores, clip_scores=None, lpips_scores=None, run_id=1, title=None):
    """
    Displays a single image slider and a table of overall metric scores.
    Optionally overlays CLIP and LPIPS scores under each image.

    Args:
        images (str): Path to the image slider (combined image strip).
        tag (str): Identifier name for this slider/model.
        logger (neptune.run.Run or None): Neptune logger instance.
        scores (dict): Dictionary of overall metric name -> float.
        clip_scores (dict[int, float] or None): Per-scale CLIP scores.
        lpips_scores (dict[int, float] or None): Per-scale LPIPS scores.
    """
    if title is None:
        title = tag

    # Load the image slider strip
    slider_img = mpimg.imread(images)

    # Create figure
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"Concept: {title}", fontsize=16, weight='bold')
    gs = gridspec.GridSpec(2, 1)

    # Top row: image slider
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(slider_img)
    ax1.axis('off')

    # Bottom row: metric score table
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('off')
    table_data = [(k, f"{v:.4f}") for k, v in scores.items()]
    col_labels = ["Metric", "Score"]

    table = ax2.table(cellText=table_data, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.5, 1.3)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save figure
    os.makedirs("outputs/sd", exist_ok=True)
    fig_path = f"outputs/sd/overall_plot_{run_id}.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)

    if logger is not None:
        logger[f"sliders_and_plots/{tag}"].log(neptune.types.File(fig_path))

    plt.close()
    return fig_path
