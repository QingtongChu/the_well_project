from the_well.utils.download import well_download

BASE_PATH = "./datasets"

def main():
    for split in ["train", "valid", "test"]:
        print(f"downloading split: {split}")
        well_download(base_path=BASE_PATH, 
                      dataset = "gray_scott_reaction_diffusion", 
                      split=split)

if __name__ == "__main__":
    main()