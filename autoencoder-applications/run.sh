
checkpoints=(  "checkpoints/model_alpha_1_beta_1_phi_0.pth" "checkpoints/model_alpha_1_beta_1_phi_1.pth"
                "checkpoints/model_alpha_1_beta_1_phi_2.pth" "checkpoints/model_alpha_1_beta_1_phi_5.pth"
                "checkpoints/model_alpha_1_beta_1_phi_10.pth" "checkpoints/model_alpha_1_beta_1_phi_15.pth"
                "checkpoints/model_alpha_1_beta_1_phi_20.pth")

for ckpt in "${checkpoints[@]}"; do
    python results_eval.py "$ckpt"
done
