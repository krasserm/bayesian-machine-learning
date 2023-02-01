
checkpoints=(  "results/checkpoints/model_alpha_1_beta_1_phi_0.pth" "results/checkpoints/model_alpha_1_beta_1_phi_1.pth"
                "results/checkpoints/model_alpha_1_beta_1_phi_2.pth" "results/checkpoints/model_alpha_1_beta_1_phi_5.pth"
                "results/checkpoints/model_alpha_1_beta_1_phi_10.pth" "results/checkpoints/model_alpha_1_beta_1_phi_15.pth"
                "results/checkpoints/model_alpha_1_beta_1_phi_20.pth")

for ckpt in "${checkpoints[@]}"; do
    python results_eval.py "$ckpt"
done
