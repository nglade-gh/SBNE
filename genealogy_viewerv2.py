import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
from get_agent_lineage_with_visualization import get_agent_lineage_with_visualization


class GenealogyViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualiseur de Généalogie")
        self.root.geometry("600x400")

        self.simulation_dir = tk.StringVar()
        self.agent_ids = tk.StringVar(value="1")
        self.max_descendants = tk.IntVar(value=5)

        self.setup_ui()

    def setup_ui(self):
        sim_frame = ttk.LabelFrame(self.root, text="Sélectionner une simulation")
        sim_frame.pack(padx=10, pady=10, fill="x")

        self.simulations = self.get_simulation_list()
        if not self.simulations:
            ttk.Label(sim_frame, text="Aucune simulation trouvée dans 'simulations/'").pack(pady=5)
            return

        self.simulation_combobox = ttk.Combobox(
            sim_frame,
            textvariable=self.simulation_dir,
            values=self.simulations,
            state="readonly"
        )
        self.simulation_combobox.pack(pady=5, fill="x")
        self.simulation_combobox.current(0)

        agent_frame = ttk.LabelFrame(self.root, text="ID(s) de l'agent (séparés par des virgules)")
        agent_frame.pack(padx=10, pady=5, fill="x")

        ttk.Entry(agent_frame, textvariable=self.agent_ids, width=50).pack(pady=5, padx=10)

        desc_frame = ttk.LabelFrame(self.root, text="Nombre max de descendants à afficher")
        desc_frame.pack(padx=10, pady=5, fill="x")

        ttk.Spinbox(
            desc_frame,
            from_=1,
            to=50,
            textvariable=self.max_descendants,
            width=10
        ).pack(pady=5)

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Analyser", command=self.analyze_genealogy).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Ouvrir le dossier", command=self.open_simulation_folder).pack(side="left", padx=5)

        self.result_text = tk.Text(self.root, height=10, wrap="word", state="disabled")
        self.result_text.pack(padx=10, pady=5, fill="both", expand=True)

    def get_simulation_list(self):
        sim_dir = "simulations"
        if not os.path.exists(sim_dir):
            return []
        return [entry.name for entry in os.scandir(sim_dir) if entry.is_dir()]

    def get_db_path(self, simulation_name):
        default_db_path = os.path.join("simulations", simulation_name, "evolution.db")
        if os.path.exists(default_db_path):
            return default_db_path

        parameters_path = os.path.join("simulations", simulation_name, "parameters.json")
        if os.path.exists(parameters_path):
            with open(parameters_path, "r") as f:
                parameters = json.load(f)
                db_path = parameters.get("DB_PATH")
                if db_path and os.path.exists(db_path):
                    return db_path
        return None

    def analyze_genealogy(self):
        simulation_name = self.simulation_dir.get()
        if not simulation_name:
            messagebox.showerror("Erreur", "Aucune simulation sélectionnée.")
            return

        raw_ids = [s.strip() for s in self.agent_ids.get().split(",")]
        agent_ids = [s for s in raw_ids if s]
        if not agent_ids:
            messagebox.showerror("Erreur", "Aucun ID d'agent saisi.")
            return

        max_descendants = self.max_descendants.get()

        db_path = self.get_db_path(simulation_name)
        if not db_path:
            self.result_text.config(state="normal")
            self.result_text.delete(1.0, "end")
            self.result_text.insert("end", f"❌ Base de données introuvable pour {simulation_name}\n")
            self.result_text.config(state="disabled")
            return

        self.result_text.config(state="normal")
        self.result_text.delete(1.0, "end")

        for agent_id_str in agent_ids:
            try:
                # CONVERSION EXPLICITE EN INT
                agent_id = int(agent_id_str)

                self.result_text.insert("end", f"🔍 Analyse de l'agent {agent_id}...\n")
                self.root.update()

                lineage = get_agent_lineage_with_visualization(
                    db_path=db_path,
                    agent_id=agent_id,
                    max_descendants=max_descendants
                )

                self.result_text.insert("end", f"   - Ascendants : {lineage['ascendants']}\n")
                self.result_text.insert(
                    "end",
                    f"   - Descendants : {lineage['descendants'][:10]}... (Total: {len(lineage['descendants'])})\n"
                )
                self.result_text.insert(
                    "end",
                    f"   - Images dans : {os.path.dirname(db_path)}/lineage_visualizations/\n\n"
                )

            except ValueError:
                self.result_text.insert("end", f"❌ ID invalide '{agent_id_str}' : doit être un nombre entier.\n\n")
            except Exception as e:
                self.result_text.insert("end", f"❌ Erreur pour l'agent {agent_id_str} : {str(e)}\n\n")

        self.result_text.config(state="disabled")

    def open_simulation_folder(self):
        simulation_name = self.simulation_dir.get()
        if not simulation_name:
            messagebox.showerror("Erreur", "Aucune simulation sélectionnée.")
            return

        folder_path = os.path.join("simulations", simulation_name)
        if os.path.exists(folder_path):
            if os.name == "nt":
                os.startfile(folder_path)
            elif os.name == "posix":
                subprocess.Popen(["xdg-open", folder_path])
        else:
            messagebox.showerror("Erreur", f"Dossier introuvable : {folder_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GenealogyViewer(root)
    root.mainloop()