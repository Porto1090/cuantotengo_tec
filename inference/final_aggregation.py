def match_front_caps_to_bottles(front_bottles, cap_data):
    front_cap_to_bottle = {}

    for cap in cap_data:
        cx1, cy1, cx2, cy2 = cap["x1"], cap["y1"], cap["x2"], cap["y2"]

        for front in front_bottles:
            if front["x1"] <= (cx1 + cx2) / 2 <= front["x2"] and front["y1"] <= (cy1 + cy2) / 2 <= front["y2"]:
                front_cap_to_bottle[(cx1, cy1, cx2, cy2)] = (front["x1"], front["y1"], front["x2"], front["y2"])
                break

    return front_cap_to_bottle


def compute_brand_counts(bottle_brand_mapping, front_cap_to_bottle, cap_counts):
    brand_totals = {}
    lane_totals = {}

    for cap_bbox, bottle_bbox in front_cap_to_bottle.items():

        new_bottle_brand_mapping = {
            tuple(list(k)[:-1]): v for k, v in bottle_brand_mapping.items()
        }

        if bottle_bbox in new_bottle_brand_mapping:
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

# NOT USED CURRENTLY
def match_gpt_output_to_list(gpt_brand, gpt_flavor, product_dict):
    # --- 1. Brand must match exactly ---
    if gpt_brand not in product_dict:
        return "Unknown Drink"

    # --- 2. Flavor must match exactly ---
    known_flavors = product_dict[gpt_brand]

    if gpt_flavor in known_flavors:
        return f"{gpt_brand} - {gpt_flavor}"

    return f"{gpt_brand} - {gpt_flavor}"