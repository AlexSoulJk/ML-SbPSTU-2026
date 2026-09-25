def group_by_samples_and_period(images_info):
    grouped_data = {}
    for image in images_info.values():
        sample_id = image["sample_id"]
        period = image["period"]
        if sample_id not in grouped_data:
            grouped_data[sample_id] = {}
        if period not in grouped_data[sample_id]:
            grouped_data[sample_id][period] = []
        grouped_data[sample_id][period].append(image)
    return grouped_data