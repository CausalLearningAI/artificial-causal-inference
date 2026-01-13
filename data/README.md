### Data ingestion guidelines
This document outlines the guidelines for preparing and structuring data for ingestion into our processing pipeline. The goal is to ensure consistency and ease of use across different experiments and datasets.

Requirements per experiment:

    - observations/source
        - file1.mp4
        - file2.mp4
        - ...
    - annotations/
        - file1.csv        # frame-level annotations aligned to file1.mp4
        - file2.csv
        - ...
    - experiment.csv*

*experiment.csv is a mandatory file containing observation-level metadata
for a single experiment (one row per observation / recording).
It must include at least:

    - observation_id
    - observation_file
    - annotation_file
    - treatment
    - effect_modifiers (e.g., annotator, date, batch, position, ...) # save column names in config
    - start_time
    - end_time
    - fps                # fps of standardized full video
    - resolution         # resolution of standardized full video
    - ...

Then we derive the following artifacts:

    - standardized full videos
    - tracking (optional)
    - video POVs (optional)
    - dataset representation
    - embeddings
    - predictions
    - ...

---------

### Raw data structure

    data/
        ants/
            v1/
                observations/
                    source/
                        file1.mp4
                        file2.mp4
                        ...
                    full/
                        file1.mp4
                        file2.mp4
                        ...
                    povs/
                        yellow/
                            file1.mp4
                            file2.mp4
                            ...
                        blue/
                            file1.mp4
                            file2.mp4
                            ... 
                        focal/
                            file1.mp4
                            file2.mp4
                            ...
                annotations/
                experiment.csv
            v2/
                observations/
                annotations/
                experiment.csv
            ...
        mice/
            v1/
                observations/
                annotations/
                experiment.csv
            v2/
                observations/
                annotations/
                experiment.csv
            ...   