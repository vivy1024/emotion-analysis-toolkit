import optuna
import yaml
import logging
import os
import copy 
from datetime import datetime

# Import the modified training function and logger setup
from .train_lstm import train_lstm_stage2 
from .utils import setup_logging, logger # Use the existing logger setup

# --- Configuration ---
BASE_CONFIG_PATH = "18_2/config_lstm_stage2.yaml"
N_TRIALS = 30 # Number of optimization trials to run (adjust as needed)
STUDY_NAME = f"lstm_stage2_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # Unique name for the study

# --- Objective Function for Optuna ---
def objective(trial: optuna.Trial):
    """Objective function for Optuna study."""
    
    # --- Load Base Configuration ---
    try:
        with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            # Use deepcopy to avoid modifying the base config dict across trials
            base_cfg = yaml.safe_load(f) 
            cfg = copy.deepcopy(base_cfg) 
    except Exception as e:
        logger.error(f"Trial {trial.number}: Failed to load base config {BASE_CONFIG_PATH}: {e}")
        # Return a very bad value to signal failure
        return -1.0 

    # --- Define Search Space and Suggest Parameters ---
    param_overrides = {}
    param_overrides['trial_id'] = trial.number # Pass trial ID for output naming
    param_overrides['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    param_overrides['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    param_overrides['dropout_lstm'] = trial.suggest_float('dropout_lstm', 0.2, 0.7)
    param_overrides['dropout_fc'] = trial.suggest_float('dropout_fc', 0.3, 0.8)
    param_overrides['lstm_hidden_size'] = trial.suggest_categorical('lstm_hidden_size', [64, 128, 256])
    param_overrides['sadness_repeats'] = trial.suggest_int('sadness_repeats', 5, 25)
    logger.info(f"--- Trial {trial.number} Starting --- Parameters: {param_overrides}")

    # --- Run Training with Suggested Parameters ---
    try:
        # Pass the base config dictionary and the overrides
        overall_f1_macro = train_lstm_stage2(cfg, param_overrides=param_overrides) 
        logger.info(f"--- Trial {trial.number} Finished --- Overall Macro F1: {overall_f1_macro:.4f}")
        
        # Handle potential NaN or Inf values if something went wrong in calculation
        if overall_f1_macro is None or not isinstance(overall_f1_macro, (int, float)) or not (0 <= overall_f1_macro <= 1):
             logger.warning(f"Trial {trial.number}: Returned invalid F1 score ({overall_f1_macro}), treating as 0.0")
             overall_f1_macro = 0.0
             
    except Exception as e:
        logger.error(f"Trial {trial.number}: Exception during training: {e}", exc_info=True)
        # Return a very bad value if training crashes
        overall_f1_macro = -1.0 
        
    # Optuna aims to maximize the returned value
    return overall_f1_macro

# --- Main HPO Execution ---
if __name__ == "__main__":
    # --- Setup Logging for HPO Script ---
    # Use the experiment name from the config or define one
    try:
        with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
             hpo_exp_name = yaml.safe_load(f).get('experiment_name', 'lstm_stage2') + "_hpo"
    except:
        hpo_exp_name = "lstm_stage2_hpo" # Fallback name
        
    log_dir = "18_2/logs" # Define log directory or get from config
    setup_logging(log_dir, hpo_exp_name) 
    logger.info(f"Starting Optuna HPO Study: {STUDY_NAME}")
    logger.info(f"Number of trials: {N_TRIALS}")
    logger.info(f"Base config: {BASE_CONFIG_PATH}")

    # --- Create and Run Optuna Study ---
    # You can use storage for persistent studies, e.g., SQLite: storage="sqlite:///hpo_study.db"
    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME) 
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=None) # timeout in seconds if needed
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during the optimization study: {e}", exc_info=True)

    # --- Print Results ---
    logger.info("Optimization finished.")
    try:
        logger.info(f"Number of finished trials: {len(study.trials)}")
        
        best_trial = study.best_trial
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"  Value (Overall Macro F1): {best_trial.value:.4f}")
        logger.info("  Best Parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
            
        # You can also find the corresponding output directory based on the trial number
        # e.g., look for "..._trial_{best_trial.number}" in the output base directory
        
    except ValueError:
         logger.warning("No trials were completed successfully.")
    except Exception as e:
         logger.error(f"Error printing study results: {e}", exc_info=True)

    logger.info(f"HPO Study '{STUDY_NAME}' complete.") 