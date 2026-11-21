datasets = [
    "BloodMNIST", 
    "RBC_Dataset", 
    "Immuno_Synapses"
]
models   = [
    "Supervised", 
    "SimCLR", 
    "MoCoV2", 
    "MoCoV3", 
    "BYOL", 
    "BarlowTwins", 
    "DINO"
]
backbones = [
    "ResNet50", 
    "ConvNeXtV2", 
    "ViTBackbone", 
    "ConvNeXt"
]    
init_types = [
    "imagenet_weights",
    "no_weights"
]
seeds = [
    "2025",
    "7",
    "42",
]

rows = []
# for bio_transf in bio_transfs:
# for seed in seeds:
for init_type in init_types:
    for dataset in datasets:
        for backbone in backbones:
            for model in models:
                rows.append((dataset, model, backbone, init_type))

file = "combos.tsv"
with open(file,"w") as out:
    for r in rows:
        out.write("\t".join(r) + "\n")
print(f"Wrote {len(rows)} combos to {file}")
