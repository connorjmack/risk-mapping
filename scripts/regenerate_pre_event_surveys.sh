#!/bin/bash
#
# Regenerate pre_event_surveys_full.csv correctly
#
# This script:
# 1. Generates per-location survey-event pairs
# 2. Validates each output
# 3. Combines them into a single file
# 4. Verifies the final result
#

set -e  # Exit on error

# Configuration
MIN_VOLUME=5.0
MIN_DAYS_BEFORE=7
LOCATIONS=(DelMar SanElijo Encinitas Solana Torrey Blacks)
OUTPUT_DIR="data"
EVENTS_DIR="utiliies/events/qc_ed"
SURVEYS_CSV="utiliies/file_lists/all_noveg_files.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Regenerating Pre-Event Survey Dataset"
echo "=========================================="
echo ""

# Check prerequisites
if [ ! -f "$SURVEYS_CSV" ]; then
    echo -e "${RED}Error: Survey catalog not found: $SURVEYS_CSV${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clean up old files
echo "Cleaning up old files..."
rm -f "$OUTPUT_DIR"/pre_event_surveys_*.csv
echo -e "${GREEN}✓ Removed old files${NC}"
echo ""

# Process each location
total_rows=0
successful_locations=0

for location in "${LOCATIONS[@]}"; do
    echo "=========================================="
    echo "Processing: $location"
    echo "=========================================="

    # Find most recent QC file for this location
    event_file=$(ls -t "$EVENTS_DIR"/${location}_events_qc_*.csv 2>/dev/null | head -1)

    if [ -z "$event_file" ]; then
        echo -e "${YELLOW}⚠ Warning: No QC file found for $location, skipping...${NC}"
        echo ""
        continue
    fi

    echo "Event file: $event_file"
    event_count=$(tail -n +2 "$event_file" | wc -l | tr -d ' ')
    echo "Events in file: $event_count"

    # Generate survey-event pairs
    output_file="$OUTPUT_DIR/pre_event_surveys_${location}.csv"

    python scripts/01_identify_surveys.py \
        --events "$event_file" \
        --surveys "$SURVEYS_CSV" \
        --output "$output_file" \
        --location "$location" \
        --min-volume "$MIN_VOLUME" \
        --min-days-before "$MIN_DAYS_BEFORE" \
        -v

    # Validate output
    if [ ! -f "$output_file" ]; then
        echo -e "${RED}✗ Failed to generate output for $location${NC}"
        echo ""
        continue
    fi

    # Check file size and row count
    file_size=$(ls -lh "$output_file" | awk '{print $5}')
    row_count=$(tail -n +2 "$output_file" | wc -l | tr -d ' ')

    echo ""
    echo "Output validation:"
    echo "  File: $output_file"
    echo "  Size: $file_size"
    echo "  Rows: $row_count"

    # Sanity check: rows should be <= events (since some events may not have pre-surveys)
    if [ "$row_count" -gt "$((event_count * 2))" ]; then
        echo -e "${RED}✗ ERROR: Row count ($row_count) is suspiciously high!${NC}"
        echo -e "${RED}  Expected <= $event_count rows (one per event)${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Validated successfully${NC}"
    echo ""

    total_rows=$((total_rows + row_count))
    successful_locations=$((successful_locations + 1))
done

# Check if we have any data
if [ "$successful_locations" -eq 0 ]; then
    echo -e "${RED}Error: No locations were processed successfully${NC}"
    exit 1
fi

echo "=========================================="
echo "Combining Location Files"
echo "=========================================="
echo "Locations processed: $successful_locations"
echo "Total rows: $total_rows"
echo ""

# Combine files
output_full="$OUTPUT_DIR/pre_event_surveys_full.csv"

# Get header from first file
first_file=$(ls "$OUTPUT_DIR"/pre_event_surveys_*.csv | head -1)
head -1 "$first_file" > "$output_full"

# Append data from all files (skip headers)
for location in "${LOCATIONS[@]}"; do
    location_file="$OUTPUT_DIR/pre_event_surveys_${location}.csv"
    if [ -f "$location_file" ]; then
        tail -n +2 "$location_file" >> "$output_full"
    fi
done

echo "Combined file: $output_full"

# Final validation
echo ""
echo "=========================================="
echo "Final Validation"
echo "=========================================="

final_size=$(ls -lh "$output_full" | awk '{print $5}')
final_rows=$(tail -n +2 "$output_full" | wc -l | tr -d ' ')

echo "File: $output_full"
echo "Size: $final_size"
echo "Rows: $final_rows"
echo ""

# Check if file is reasonable size (should be < 100 MB)
file_size_bytes=$(stat -f%z "$output_full" 2>/dev/null || stat -c%s "$output_full")
max_size=$((100 * 1024 * 1024))  # 100 MB

if [ "$file_size_bytes" -gt "$max_size" ]; then
    echo -e "${RED}✗ ERROR: File is too large ($final_size)!${NC}"
    echo -e "${RED}  Expected < 100 MB. Something went wrong.${NC}"
    exit 1
fi

# Check row count is reasonable (should be < 50,000 for these beaches)
if [ "$final_rows" -gt 50000 ]; then
    echo -e "${YELLOW}⚠ WARNING: Row count seems high ($final_rows)${NC}"
    echo -e "${YELLOW}  Expected < 50,000. Please verify manually.${NC}"
else
    echo -e "${GREEN}✓ Row count looks reasonable${NC}"
fi

if [ "$file_size_bytes" -lt "$((10 * 1024 * 1024))" ]; then
    echo -e "${GREEN}✓ File size looks reasonable${NC}"
fi

echo ""
echo "=========================================="
echo "SUCCESS"
echo "=========================================="
echo "Generated: $output_full"
echo "Total rows: $final_rows"
echo "File size: $final_size"
echo ""
echo "Next step:"
echo "python scripts/04_assemble_training_data.py \\"
echo "    --features data/polygon_features_fullscale.csv \\"
echo "    --surveys data/pre_event_surveys_full.csv \\"
echo "    --output data/training_data.csv -v"
