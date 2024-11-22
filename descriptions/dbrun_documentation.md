# Goal

Provide a runner for reproducible, self documenting pipelines. The code and most basic information can be accessed on [github](https://github.com/JulienBrn/database_scripts).

# Usage

(Assuming dbrun is installed on this computer)

The command line is `dbrun <path_to_your_pipeline_file>`, so if your pipeline file is located at */home/t4user/Documents/PipelineConfigs/BirdDurationShift/create_labelling_model.yaml*, the command will be `dbrun '/home/t4user/Documents/PipelineConfigs/BirdDurationShift/create_labelling_model.yaml'`.

Notes:
- `dbrun` should be executed on this computer. You may ssh to it before hand if you wish to run it from elsewhere
- `dbrun` can be executed from any conda environment or current directory. Note that it may require running from a zsh terminal.
- The `'` quotes are here to help the terminal read the path correctly, especially if it has spaces or special characters.
- Your *path_to_your_pipeline_file* can either be a valid path in the terminal from which you launch that command or a link to a file on the web.
  Note that if you provide a link, the path in that file will still be interpreted either as path on this computer or as full links, not as relative links.

# Writting the pipeline file


# Installing dbrun on a computer



