from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion


def main():
    pipeline = OmniDiffusion(model="../models/BAGEL-7B-MoT")
    result = pipeline.generate("A cute cat", seed=52)
    result.images[0].save("bagel_i2i_output.png")


if __name__ == "__main__":
    main()
