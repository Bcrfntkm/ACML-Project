#!/usr/bin/env bash
# Download open-access chemistry papers from PubMed Central.
# All papers: synthesis of Cu/Ni/Co/Zn/Mn coordination complexes — same domain.
# Usage: bash Science/scripts/download_papers.sh [OUT_DIR]

OUT_DIR="${1:-Science/data/pdfs}"
mkdir -p "$OUT_DIR"

download_pmc() {
    local id="$1"
    local dest="$OUT_DIR/${id}.pdf"
    if [ -f "$dest" ] && [ "$(wc -c < "$dest")" -gt 10000 ]; then
        echo "[SKIP] $id (already exists)"
        return
    fi
    local url="https://pmc.ncbi.nlm.nih.gov/articles/${id}/pdf"
    echo "[DOWNLOAD] $id"
    curl -L -s -o "$dest" "$url" \
        -H "User-Agent: Mozilla/5.0 (compatible; academic-research)" \
        --retry 3 --retry-delay 2
    local size
    size=$(wc -c < "$dest" 2>/dev/null || echo 0)
    if [ "$size" -lt 10000 ]; then
        echo "[WARN] $id may be invalid ($size bytes)"
        rm -f "$dest"
    else
        echo "[OK] $id ($(du -h "$dest" | cut -f1))"
    fi
}

# Copper / Nickel / Cobalt / Zinc Schiff-base complexes — synthesis & characterisation
download_pmc PMC6272500
download_pmc PMC8173565
download_pmc PMC11130647
download_pmc PMC10328999
download_pmc PMC9159852
download_pmc PMC10142776
download_pmc PMC8510892
download_pmc PMC9043818
download_pmc PMC9669718
download_pmc PMC3269312
download_pmc PMC10882688
download_pmc PMC10447992
download_pmc PMC11416489
download_pmc PMC10324070
download_pmc PMC11500387

echo ""
echo "=== Done. PDFs in $OUT_DIR ==="
ls "$OUT_DIR"/*.pdf 2>/dev/null | while read f; do
    printf "  %-40s %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
done
