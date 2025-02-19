#!/bin/bash

# Pfad zum Zielordner
destination="all_features"

# Zielordner erstellen, falls er nicht existiert
mkdir -p "$destination"

# Durchsuchen aller Ordner mit dem Muster "Drug*_Analysis"
for dir in Drug*_analysis; 
do
    if [[ -d "$dir/best" ]]; then
        # Extrahiere den Namen des Drug-Ordners
        drug_name=$(basename "$dir" | cut -d'_' -f1)
        
        # Pfad zur features_0 Datei
        source_file="$dir/best/features_0.csv"
        
        # Überprüfen, ob die Datei existiert
        if [[ -f "$source_file" ]]; then
            # Zielpfad und neuer Dateiname
            destination_file="$destination/${drug_name}_features.csv"
            
            # Datei kopieren und umbenennen
            cp "$source_file" "$destination_file"
            echo "Kopiert: $source_file nach $destination_file"
        else
            echo "Warnung: Datei $source_file existiert nicht"
        fi
    else
        echo "Warnung: Ordner $dir/best existiert nicht"
    fi
done

echo "Fertig!"

