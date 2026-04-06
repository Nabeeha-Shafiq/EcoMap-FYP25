import yaml
import sys
from pathlib import Path

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Build teacher config
teacher_config_path = str(Path(config_file).parent / "modular_teacher_extract_TEMP.yaml")

# Get teacher output dir
teacher_output_dir = config.get('teacher', {}).get('output_dir', './results')

teacher_section = {
    'dataset': config.get('dataset', {}),
    'input_dataset': config.get('input_dataset', {}),
    'embeddings': config.get('embeddings', {}),
    'preprocessing': config.get('preprocessing', {}),
    'pipeline': {k: v for k, v in config.get('pipeline', {}).items() 
                 if k in ['n_folds', 'folds_test_size', 'random_seed', 'n_jobs', 'device']},
    'output': {  # OLD pipeline expects 'output' section
        'output_dir': teacher_output_dir
    },
    'training': config.get('training', {}),
    'teacher': config.get('teacher', {}),
    'logging': config.get('logging', {})
}

with open(teacher_config_path, 'w') as f:
    yaml.dump(teacher_section, f, default_flow_style=False)

# Build student config
student_config_path = str(Path(config_file).parent / "modular_student_extract_TEMP.yaml")
student_section = dict(config.get('student', {}))

teacher_output = config.get('teacher', {}).get('output_dir', '')
student_output_dir = config.get('student', {}).get('output_dir', './results')

# Extract training parameters FROM student section
training_config = {
    'learning_rate': student_section.get('learning_rate', 0.001),
    'batch_size': student_section.get('batch_size', 32),
    'n_epochs': student_section.get('n_epochs', 150),
    'weight_decay': student_section.get('weight_decay', 1e-5),
    'dropout_rate': student_section.get('dropout_rate', 0.3),
    'use_batch_norm': student_section.get('use_batch_norm', True),
    'use_class_weights': student_section.get('use_class_weights', True),
    'early_stopping_patience': student_section.get('early_stopping_patience', 15),
    'validation_split': student_section.get('validation_split', 0.2),
    'n_folds': config.get('pipeline', {}).get('n_folds', 5),
    'random_seed': config.get('pipeline', {}).get('random_seed', 42),
    'hidden_dims': student_section.get('hidden_dims', [128, 64, 32])
}

# Extract distillation parameters FROM student.distillation section
distillation_config = dict(student_section.get('distillation', {}))
if teacher_output:
    distillation_config['teacher_model_path'] = f"{teacher_output}/teacher_model_FROZEN.pt"
    distillation_config['teacher_preprocessed_dir'] = f"{teacher_output}/.working/preprocessed_arrays"

# Set defaults for distillation if not provided
distillation_config.setdefault('alpha', 0.7)
distillation_config.setdefault('beta', 0.2)
distillation_config.setdefault('gamma', 0.1)
distillation_config.setdefault('temperature', 4.0)

# Handle student-specific embeddings (for ablation studies)
student_input_dataset = dict(config.get('input_dataset', {}))
if 'student_embeddings' in config:
    # Override with student-specific embeddings if provided (for cell-only, gene-only ablations)
    student_embeddings = config['student_embeddings']
    if 'image_encoder_embeddings' in student_embeddings:
        student_input_dataset['image_encoder_embeddings'] = student_embeddings['image_encoder_embeddings']
    if 'gene_embeddings' in student_embeddings:
        student_input_dataset['gene_embeddings'] = student_embeddings['gene_embeddings']
    if 'cell_composition_embeddings' in student_embeddings:
        student_input_dataset['cell_composition_embeddings'] = student_embeddings['cell_composition_embeddings']

student_full = {
    'dataset': config.get('dataset', {}),
    'input_dataset': student_input_dataset,
    'embeddings': config.get('embeddings', {}),
    'preprocessing': config.get('preprocessing', {}),
    'pipeline': {k: v for k, v in config.get('pipeline', {}).items() 
                 if k in ['n_folds', 'folds_test_size', 'random_seed', 'n_jobs', 'device']},
    'output': {  # OLD pipeline expects 'output' section
        'output_dir': student_output_dir
    },
    'training': training_config,
    'distillation': distillation_config,
    'student': student_section,
    'logging': config.get('logging', {})
}

with open(student_config_path, 'w') as f:
    yaml.dump(student_full, f, default_flow_style=False)

# Output the paths
print(teacher_config_path)
print(student_config_path)
print(teacher_output)
