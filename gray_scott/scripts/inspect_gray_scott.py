import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from the_well.data import WellDataset

BASE_PATH = "./datasets"
DATASET_NAME = "gray_scott_reaction_diffusion"

def main():
    #1. Load the training data print name and number of samples
    dataset = WellDataset(well_base_path=BASE_PATH,
                          well_dataset_name=DATASET_NAME,
                          well_split_name="train",
                          n_steps_input=8,
                          n_steps_output=1,
                          use_normalization=False)
    print(f"Loaded dataset: {DATASET_NAME}")
    print(f"Number of samples: {dataset.len}")

    #2. Inspect one sample. First get the keys in dict, then the shape of input output tensor
    sample = dataset[100]
    print(f"Item keys: {list(sample.keys())}")
    input = sample["input_fields"]
    output = sample["output_fields"]
    print(f"Shape of input tensor: {input.shape}")
    print(f"Shape of output tensor: {output.shape}")

    #3. Show the metadata of the about the fields. Count and names
    F = dataset.metadata.n_fields
    field_names = [name for group in dataset.metadata.field_names.values() for name in group]
    print(f"Total number of fields we have: ", F)
    print(dataset.metadata.field_names)
    print("Names of fields: ", field_names)

    #4. Visualize one sample
    x = rearrange(input, "T Lx Ly F -> F T Lx Ly")
    T  = x.shape[1]
    fig, axs = plt.subplots(F, T, figsize=(2 * T, 2.4 * F))

    if F == 1:
        axs = np.expand_dims(axs, axis=0)

    for f in range(F):
        vmin = np.nanmin(x[f])
        vmax = np.nanmax(x[f])
        for t in range(8):
            ax = axs[f, t]
            ax.imshow(x[f, t], cmap="RdBu_r", interpolation="none", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

            if f == 0:
                ax.set_title(f"t = {t}")
        
        label = field_names[f] if f < len(field_names) else f"field {f}"
        axs[f, 0].set_ylabel(label)
        
    plt.tight_layout()
    plt.show()
        


if __name__ == "__main__":
    main()