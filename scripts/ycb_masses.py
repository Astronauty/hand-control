"""Published YCB object masses, transcribed from Table II of the YCB paper.

    B. Calli, A. Singh, J. Bruce, A. Walsman, K. Konolige, S. Srinivasa,
    P. Abbeel, A. M. Dollar, "Yale-CMU-Berkeley dataset for robotic manipulation
    research", and "The YCB Object and Model Set and Benchmarking Protocols",
    IEEE Robotics & Automation Magazine.  https://www.ycbbenchmarks.com/

Masses are measured with a digital scale to +-1 g. The set publishes mass and
principal dimensions -- it does **not** publish inertia tensors. So the tensor
MuJoCo gets is computed from each object's own sealed geometry assuming uniform
density, scaled so the total equals the published mass. The mass is therefore
exactly the published figure; the tensor is as good as the uniform-density
assumption, which is the usual one for scanned rigid objects.

Why this matters at all: with no explicit <inertial>, MuJoCo infers mass from the
geoms, which counts the visual mesh *and* every overlapping collision hull. On
011_banana that gave 0.379 kg against a published 0.066 kg -- 5.7x heavy, and the
error moved whenever the hull count changed.

Directory prefixes are YCB object IDs, so the ID is noted against each entry.
"""

# Objects the table names individually. Directory name -> grams.
MASS_G = {
    "002_master_chef_can": 414,     # 2
    "003_cracker_box": 411,         # 3
    "004_sugar_box": 514,           # 4
    "005_tomato_soup_can": 349,     # 5
    "006_mustard_bottle": 603,      # 6
    "007_tuna_fish_can": 171,       # 7
    "008_pudding_box": 187,         # 8
    "009_gelatin_box": 97,          # 9
    "010_potted_meat_can": 370,     # 10
    "011_banana": 66,               # 11
    "012_strawberry": 18,           # 12
    "013_apple": 68,                # 13
    "014_lemon": 29,                # 14
    "015_peach": 33,                # 15
    "016_pear": 49,                 # 16
    "017_orange": 47,               # 17
    "018_plum": 25,                 # 18
    "019_pitcher_base": 178,        # 19
    "021_bleach_cleanser": 1131,    # 21
    "022_windex_bottle": 1022,      # 22
    "024_bowl": 147,                # 24
    "025_mug": 118,                 # 25
    "026_sponge": 6.2,              # 26
    "027_skillet": 950,             # 27
    "027-a_skillet": 950,           # 27, same skillet rescanned: extents match
                                    #     (267x441x142 vs 461x266x139, axes permuted)
    "028_skillet_lid": 652,         # 28
    "029_plate": 279,               # 29
    "030_fork": 34,                 # 30
    "031_spoon": 30,                # 31
    "032_knife": 31,                # 32
    "033_spatula": 51.5,            # 33
    "035_power_drill": 895,         # 35
    "036_wood_block": 729,          # 36
    "037_scissors": 82,             # 37
    "037-a_scissors": 82,           # 37, identical extents to 037 (96.1x201.5x15.7)
    "037-b_scissors": 82,           # 37, same scissors opened out
    "038_padlock": 304,             # 38
    "040_large_marker": 15.8,       # 40
    "042_adjustable_wrench": 252,   # 42
    "043_phillips_screwdriver": 97,  # 43
    "044_flat_screwdriver": 98.4,   # 44
    "048_hammer": 665,              # 48
    "050_medium_clamp": 59,         # 50  (M Clamp)
    "051_large_clamp": 125,         # 51  (L Clamp)
    "052_extra_large_clamp": 202,   # 52  (XL Clamp)
    "053_mini_soccer_ball": 123,    # 53
    "054_softball": 191,            # 54
    "055_baseball": 148,            # 55
    "056_tennis_ball": 58,          # 56
    "057_racquetball": 41,          # 57
    "058_golf_ball": 46,            # 58
    "059_chain": 98,                # 59
    "061_foam_brick": 28,           # 61
    "062_dice": 5.2,                # 62
    # 65: the table lists the ten stacking cups as one bracketed series,
    # [13,14,17,19,21,26,28,31,33.5,38] g, ascending with cup size a..j.
    "065-a_cups": 13,
    "065-b_cups": 14,
    "065-c_cups": 17,
    "065-d_cups": 19,
    "065-e_cups": 21,
    "065-f_cups": 26,
    "065-g_cups": 28,
    "065-h_cups": 31,
    "065-i_cups": 33.5,
    "065-j_cups": 38,
}

# Objects the table gives only a *combined* figure for: the scans are individual
# pieces, the published number covers the whole assembly. Splitting it across the
# pieces by sealed volume assumes one material throughout, which holds well for
# the Lego and the peg-test board and less well for the airplane, but it is still
# published data rather than an invented density. Prefix -> grams for the set.
ASSEMBLY_TOTAL_G = {
    "071": 1435,    # 9-Peg-Hole Test: board plus its pegs
    "072": 570,     # Toy Airplane: assembled from its parts
    "073": 523,     # Lego Duplo: the set
}

# Not in the table at all: marbles are listed "N/A", and the Rubik's cube (077)
# postdates it. These fall back to a nominal density, and are reported as such.
NOMINAL_DENSITY = 1000.0    # kg/m^3, water


def resolve(name, sealed_volume_m3, assembly_volumes_m3=None):
    """Return (mass_kg, source) for one object.

    `assembly_volumes_m3` maps sibling directory name -> sealed volume, and is
    required only for the assembly groups above.
    """
    if name in MASS_G:
        return MASS_G[name] / 1000.0, "published"

    prefix = name.split("-")[0].split("_")[0]
    if prefix in ASSEMBLY_TOTAL_G and assembly_volumes_m3:
        total = sum(assembly_volumes_m3.values())
        if total > 0:
            share = sealed_volume_m3 / total
            return ASSEMBLY_TOTAL_G[prefix] / 1000.0 * share, "assembly_share"

    return NOMINAL_DENSITY * sealed_volume_m3, "nominal_density"


def group_prefix(name):
    """Assembly group a name belongs to, or None."""
    prefix = name.split("-")[0].split("_")[0]
    return prefix if prefix in ASSEMBLY_TOTAL_G else None
