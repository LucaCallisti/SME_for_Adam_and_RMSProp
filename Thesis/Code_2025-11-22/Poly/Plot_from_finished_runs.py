import wandb
import os
import torch

def import_file_from_wandb(run_id, project, entity, name_to_search):
    api = wandb.Api()
    my_run = api.run(f"{entity}/{project}/{run_id}")
    
    for artifact in my_run.logged_artifacts():
        if name_to_search in artifact.name:
            target_file = None
        
        # Scorriamo i file contenuti nell'artefatto
        for file in artifact.files():
            if file.name.endswith(".pt"):
                target_file = file.name
                break 
        
        if target_file:
            print(f"Scaricando il file specifico: {target_file}")
            
            # Usiamo il nome del file reale, non il nome dell'artefatto
            file_path = artifact.get_path(target_file).download()
            
            # Carichiamo il dizionario
            my_dict = torch.load(file_path, map_location='cpu')
            
            print(f"Dizionario caricato per run {run_id}.")
            print(f"Chiavi disponibili: {my_dict.keys()}")
        else:
            print(f"Attenzione: Nessun file .pt trovato dentro {artifact.name}")
        # ------------------------
        
        break
    return my_dict

def download_wandb_run_files(run_id1, run_id2, project, entity):
    dict_run_1 = import_file_from_wandb(run_id1, project, entity, 'final_results')
    dict_run_2 = import_file_from_wandb(run_id2, project, entity, 'final_results')

    needed_keys = ['initial_points_before_disc', 'initial_points_after_disc', 'final_time']
    tau = dict_run_1['tau']
    result_dir = "downloaded_results"
    os.makedirs(result_dir, exist_ok=True)
    final_results = { key : dict_run_1[key] for key in needed_keys }
    final_results['2_order_Balistic'] = dict_run_1['2_order_Balistic']
    final_results['2_order_BatchEq'] = dict_run_2['2_order_BatchEq']

    

    breakpoint()
    
       


if __name__ == "__main__":
    # Esempio di utilizzo
    run_id1 = "yl6bkgpo"
    run_id2 = "z63ekf3v"
    project = "Poly_with_additional_noise"
    entity = "Effective-continuous-equations"
    target_dir = "wandb_downloads"
    file_names = ["model.pth", "metrics.json"]  # Specifica i file da scaricare o None per tutti

    download_wandb_run_files(run_id1, run_id2, project, entity, target_dir, file_names)