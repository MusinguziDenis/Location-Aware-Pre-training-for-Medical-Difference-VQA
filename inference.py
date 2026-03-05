from diff_vlm import DiffVLM, VLMConfig
import argparse

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from main import vlm_processors

def main():
    parser = argparse.ArgumentParser(description="Pass arguments for inference")
    
    # Add arguments
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint for loading weights",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The text prompt for generation",
    )

    parser.add_argument(
        "--main-image-path",
        type=str,
        required=True,
        help="Path to the main image",
    )

    parser.add_argument(
        "--ref-image-path",
        type=str,
        required=True,
        help="Path to the reference image",
    )

    parser.add_argument(
        "--max-tokens-to-generate",
        type=int,
        default=64,
        help="Maximum number of tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling",
    )

    args = parser.parse_args()

    # Access them
    model_checkpoint = args.model_checkpoint
    prompt = args.prompt
    main_image_path = args.main_image_path
    ref_image_path = args.ref_image_path
    max_tokens_to_generate = args.max_tokens_to_generate
    temperature = args.temperature
    top_k = args.top_k

    # Load model
    vlm_config = VLMConfig()
    vlm = DiffVLM(vlm_config)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    vlm.load_state_dict(checkpoint["model_state_dict"])
    vlm = vlm.to(device)
    vlm.eval()

    # Generate response
    response = vlm.generate(
        processor=vlm_processors,
        prompt=prompt,
        main_image_path=main_image_path,
        ref_image_path=ref_image_path,
        max_tokens_to_generate=max_tokens_to_generate,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
    )

    print("Generated Response:")
    print(response)

if __name__ == "__main__":
    main()