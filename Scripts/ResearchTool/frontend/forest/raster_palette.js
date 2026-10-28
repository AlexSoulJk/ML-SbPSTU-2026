function hexToRgb(color) {
  const value = String(color || "").replace("#", "");
  if (value.length !== 6) return null;
  const rgb = [
    Number.parseInt(value.slice(0, 2), 16),
    Number.parseInt(value.slice(2, 4), 16),
    Number.parseInt(value.slice(4, 6), 16),
  ];
  return rgb.some((channel) => Number.isNaN(channel)) ? null : rgb;
}

export function createPaletteClassifier(legend, keyField) {
  const entries = (legend || [])
    .map((item) => {
      const rgb = hexToRgb(item.color);
      return rgb ? { key: String(item[keyField]), rgb } : null;
    })
    .filter(Boolean);
  const exact = new Map(
    entries.map((entry) => [`${entry.rgb[0]},${entry.rgb[1]},${entry.rgb[2]}`, entry.key]),
  );

  return (red, green, blue) => {
    const exactKey = exact.get(`${red},${green},${blue}`);
    if (exactKey !== undefined) return exactKey;

    let closestKey = null;
    let closestDistance = Number.POSITIVE_INFINITY;
    entries.forEach((entry) => {
      const redDelta = red - entry.rgb[0];
      const greenDelta = green - entry.rgb[1];
      const blueDelta = blue - entry.rgb[2];
      const distance = redDelta * redDelta + greenDelta * greenDelta + blueDelta * blueDelta;
      if (distance < closestDistance) {
        closestDistance = distance;
        closestKey = entry.key;
      }
    });
    return closestKey;
  };
}
