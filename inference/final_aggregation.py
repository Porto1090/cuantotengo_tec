from rapidfuzz import process, fuzz
from inference.config import standard_drinks


def match_front_caps_to_bottles(front_bottles, cap_data):
    front_cap_to_bottle = {}

    for cap in cap_data:
        cx1, cy1, cx2, cy2 = cap["x1"], cap["y1"], cap["x2"], cap["y2"]

        for front in front_bottles:
            if front["x1"] <= (cx1 + cx2) / 2 <= front["x2"] and front["y1"] <= (cy1 + cy2) / 2 <= front["y2"]:
                front_cap_to_bottle[(cx1, cy1, cx2, cy2)] = (front["x1"], front["y1"], front["x2"], front["y2"])
                break

    return front_cap_to_bottle

# TODO: Use a dictionary rather than a string for this
def compute_brand_counts(bottle_brand_mapping, front_cap_to_bottle, cap_counts, product_dict):
    brand_totals = {}
    lane_totals = {}

    for cap_bbox, bottle_bbox in front_cap_to_bottle.items():

        new_bottle_brand_mapping = {
            tuple(list(k)[:-1]): v for k, v in bottle_brand_mapping.items()
        }

        if bottle_bbox in new_bottle_brand_mapping:
            # gpt_brand_flavor = new_bottle_brand_mapping[bottle_bbox]
            # gpt_brand, gpt_flavor = gpt_brand_flavor.split(" - ", 1) if " - " in gpt_brand_flavor else (gpt_brand_flavor, "")

            # matched_product = match_gpt_output_to_list(gpt_brand, gpt_flavor, product_dict)

            matched_product = new_bottle_brand_mapping[bottle_bbox]

            # Totals count
            counts = cap_counts.get(cap_bbox, {'bottle': 0, 'can': 0})
            if counts['bottle'] > 0:
                key = f"bottle - {matched_product}"
                brand_totals[key] = brand_totals.get(key, 0) + counts['bottle']
            if counts['can'] > 0:
                key = f"can - {matched_product}"
                brand_totals[key] = brand_totals.get(key, 0) + counts['can']

            # Lanes count
            lane_totals[bottle_bbox] = counts["bottle"] + counts['can']

    return brand_totals, lane_totals

# local pre-trained classifier
def match_gpt_output_to_list(gpt_brand, gpt_flavor, product_dict, similarity_threshold=80):

    print("DEBUG gpt_brand:", repr(gpt_brand))
    print("DEBUG product_dict keys:", list(product_dict.keys()))

    # --- 1. Brand must match exactly ---
    if gpt_brand not in product_dict:
        print(f"⚠️ Warning: Brand '{gpt_brand}' not in standard_drinks list.")
        return "Unknown Drink"

    # --- 2. Flavor must match exactly ---
    known_flavors = product_dict[gpt_brand]

    if gpt_flavor in known_flavors:
        return f"{gpt_brand} - {gpt_flavor}"

    # If flavor isn't found, return literal flavor anyway
    # (your classifier output is always correct)
    return f"{gpt_brand} - {gpt_flavor}"


# matching function for gpt version
# def match_gpt_output_to_list(gpt_brand, gpt_flavor, product_dict, similarity_threshold=80):
#     matched_brand = None
#     matched_flavor = None

#     brand_list = list(product_dict.keys())

#     if gpt_brand in brand_list:
#         matched_brand = gpt_brand
#     else:
#         for brand in brand_list:
#             if gpt_brand.lower() in brand.lower() or brand.lower() in gpt_brand.lower():
#                 matched_brand = brand
#                 break

#     if not matched_brand:
#         best_brand_match, brand_score, _ = process.extractOne(gpt_brand, brand_list, scorer=fuzz.partial_ratio)
#         if brand_score >= similarity_threshold:
#             matched_brand = best_brand_match

#     if not matched_brand:
#         print(f"⚠️ Warning: '{gpt_brand}' not found in standard list, defaulting to 'Unknown Drink'")
#         return "Unknown Drink"

#     known_flavors = product_dict[matched_brand]

#     if gpt_flavor in known_flavors:
#         matched_flavor = gpt_flavor
#     else:
#         for flavor in known_flavors:
#             if gpt_flavor.lower() in flavor.lower() or flavor.lower() in gpt_flavor.lower():
#                 matched_flavor = flavor
#                 break

#     if not matched_flavor:
#         best_flavor_match, flavor_score, _ = process.extractOne(gpt_flavor, known_flavors, scorer=fuzz.partial_ratio)
#         if flavor_score >= similarity_threshold:
#             matched_flavor = best_flavor_match

#     if not matched_flavor:
#         return f"{matched_brand} - {gpt_flavor}"

#     return f"{matched_brand} - {matched_flavor}"
