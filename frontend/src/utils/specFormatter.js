/**
 * Specification formatting utilities for electronics catalogue and recommendation cards.
 * Prevents malformed specification labels (e.g. 'size in: 3.8', 'refresh hz: 75')
 * and duplicate units.
 */

export const SPEC_LABEL_MAP = {
  size_in: 'Screen size',
  screen_in: 'Screen size',
  resolution: 'Resolution',
  refresh_hz: 'Refresh rate',
  panel: 'Panel type',
  ram_gb: 'RAM',
  storage_gb: 'Storage',
  cpu: 'Processor',
  gpu: 'Graphics',
  vram_gb: 'VRAM',
  battery_hr: 'Battery runtime',
  battery_mah: 'Battery capacity',
  weight_kg: 'Weight',
  weight_g: 'Weight',
  anc: 'ANC',
  stylus_support: 'Stylus support',
  ports: 'Ports',
  features: 'Features',
  water_resistance: 'Water resistance',
  driver_mm: 'Driver size',
  connectivity: 'Connectivity',
  // Phone / tablet / headphone keys that previously fell through to the
  // generic title-caser and rendered as "Camera Mp" and "Os".
  camera_mp: 'Camera',
  chipset: 'Chipset',
  os: 'OS',
  type: 'Type',
  codec: 'Codec',
  wireless: 'Wireless',
  size: 'Size',
};

export function getSpecLabel(key) {
  if (SPEC_LABEL_MAP[key]) return SPEC_LABEL_MAP[key];
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function formatSpecValue(key, val) {
  if (val == null || val === '') return 'Not verified in catalogue';
  if (Array.isArray(val)) return val.length ? val.join(', ') : 'None';
  if (typeof val === 'boolean') return val ? 'Yes' : 'No';

  const strVal = String(val).trim();

  switch (key) {
    case 'size_in':
    case 'screen_in': {
      if (
        strVal.endsWith('"') ||
        strVal.toLowerCase().endsWith('in') ||
        strVal.toLowerCase().endsWith('inch') ||
        strVal.toLowerCase().endsWith('inches')
      ) {
        return strVal;
      }
      return `${strVal}"`;
    }
    case 'refresh_hz': {
      if (strVal.toLowerCase().endsWith('hz')) return strVal;
      return `${strVal} Hz`;
    }
    case 'ram_gb':
    case 'vram_gb': {
      if (strVal.toLowerCase().endsWith('gb')) return strVal;
      return `${strVal} GB`;
    }
    case 'storage_gb': {
      if (strVal.toLowerCase().endsWith('gb') || strVal.toLowerCase().endsWith('tb')) return strVal;
      const num = Number(strVal);
      if (!isNaN(num) && num >= 1024) {
        return `${num / 1024} TB`;
      }
      return `${strVal} GB`;
    }
    case 'battery_hr': {
      if (strVal.toLowerCase().includes('hr') || strVal.toLowerCase().includes('hour')) return strVal;
      return `${strVal} hrs`;
    }
    case 'camera_mp': {
      if (strVal.toLowerCase().endsWith('mp')) return strVal;
      const mp = Number(strVal);
      return `${!isNaN(mp) && Number.isInteger(mp) ? mp : strVal} MP`;
    }
    case 'battery_mah': {
      if (strVal.toLowerCase().endsWith('mah')) return strVal;
      return `${strVal} mAh`;
    }
    case 'weight_kg': {
      if (strVal.toLowerCase().endsWith('kg')) return strVal;
      return `${strVal} kg`;
    }
    case 'weight_g': {
      if (strVal.toLowerCase().endsWith('g')) return strVal;
      return `${strVal} g`;
    }
    case 'anc': {
      return val ? 'Active Noise Cancellation' : 'Passive Isolation Only';
    }
    case 'stylus_support': {
      return val ? 'Supported' : 'Not supported';
    }
    default:
      return strVal;
  }
}
