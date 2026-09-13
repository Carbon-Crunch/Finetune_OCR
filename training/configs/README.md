# Training Configurations & Document Schema Registry

This directory contains the pipeline configurations, validation thresholds, and domain-specific JSON extraction boilerplates for industrial document OCR.

---

## 📂 Directory Layout

```
configs/
├── README.md                      # This file
├── config.yaml                    # Global preprocessing & heuristic parameters
├── generic_fallback.json          # Fallback JSON extraction schema
└── boilerplates/                  # Categorized target schema templates
    ├── registry.json              # Central document registry and classification mapping
    ├── energy/                    # Energy generation, transmission, & billing templates
    │   ├── electricity_bill.json
    │   ├── electric_consumption_history.json
    │   ├── electric_meter_reading.json
    │   ├── energy_purchase_invoice.json
    │   ├── generation_report.json
    │   ├── hand_written_power_log.json
    │   ├── hydel_bill_summary.json
    │   ├── power_plant_report.json
    │   └── steam_distribution_log.json
    ├── fuel/                      # Fuel purchase, testing, and consumption templates
    │   ├── coal_consignment_report.json
    │   ├── coal_quality_summary.json
    │   ├── coal_receipt_quality_register.json
    │   ├── daily_fuel_consumption.json
    │   ├── fuel_lab_test_report.json
    │   ├── fuel_purchase_invoice.json
    │   ├── fuel_stock_register.json
    │   ├── gas_bill.json
    │   └── rdf_waste_fuel_sampling.json
    └── production/                # Production & plant operational log templates
        ├── boiler_log_book.json
        ├── process_log_book.json
        ├── production_summary.json
        └── turbine_capacity_log.json
```

---

## ⚙️ Configuration File (`config.yaml`)

`config.yaml` controls runtime heuristics across preprocessing, document type detection, and rate limits:

### Key Sections:
1. **`preprocessing`**:
   - `ideal_min_width: 1600`
   - Blur thresholds (`blur_threshold_low: 80`, `blur_threshold_high: 300`)
   - Noise, shadow, low contrast, and skew thresholds (`skew_threshold: 1.5` degrees).
2. **`handwriting_detector`**:
   - Stroke width, curvature, slant, and loop ratio heuristics used to route documents with handwritten entries to enhanced extraction prompts.
3. **`pdf_analyzer`**:
   - Differentiates text-native PDFs from scanned image-based PDFs, table-heavy files, and mixed layouts.
4. **`content_classifier`**:
   - Detects UI noise markers (e.g. WhatsApp screenshots, camera capture banners) and filters invalid files.
5. **`file_limits`**:
   - Maximum upload size (`20 MB`) and allowed file extensions (`.jpeg`, `.jpg`, `.png`, `.pdf`).

---

## 📋 Boilerplate Registry (`boilerplates/registry.json`)

`registry.json` defines the taxonomy of supported industrial documents, organized into 3 primary data types:

| Data Type | Description | Sample Categories |
| :--- | :--- | :--- |
| **Energy** | Energy consumption, power plant generation, and utility invoices | Electricity Bill, Electric Meter Reading, Power Plant Report, Hand Written Power Log, Steam Distribution Log |
| **Fuel** | Fuel purchasing, laboratory analysis, and stock registers | Coal Consignment Report, Fuel Lab Test Report, Coal Quality Summary, Gas Bill, RDF Waste Sampling |
| **Production & Operational** | Factory logs, boiler monitoring, and turbine metrics | Boiler Log Book, Process Log Book, Turbine Capacity Log, Production Summary |

### Schema Structure
Each boilerplate JSON file defines the expected key-value structure and nesting for extractions. For example:
- **`extracted_fields`**: Key-value pairs containing `value` and optional `unit` (e.g. `{"value": "4520", "unit": "kWh"}`).
- **Tabular entries**: Array of timestamped or sequential records (e.g. `entries: [{"timestamp": "...", "temperature": {"value": "...", "unit": "C"}}]`).

---

## 🛡️ Generic Fallback (`generic_fallback.json`)

When an uploaded document cannot be confidently classified into one of the specialized registry categories, the extractor falls back to `generic_fallback.json`. This extracts broad metadata:
- Document Title / Header
- Issuer & Recipient
- Dates & Period
- Line Items / Key-Value table
- Grand Total Amount & Currency
