#!/bin/bash

# =============================================================================
# GEMMA 3-12B FINE-TUNING WITH MAXTEXT ON CHARTQA DATASET
# =============================================================================
# 
# End-to-end pipeline for fine-tuning Gemma 3-12B using MaxText and XPK on TPUs
# This script performs the following steps:
# 1. Setup Google Cloud services and infrastructure
# 2. Convert HuggingFace model to MaxText format
# 3. Test model conversion with inference
# 4. Fine-tune on ChartQA dataset using SFT
# 5. Test fine-tuned model
# 6. Deploy model for serving with JetStream
# 7. Convert back to HuggingFace format for export
#
# Using Google's AI Hypercomputer with TPU acceleration
#
# Copyright 2024 Google, LLC. This software is provided as-is,
# without warranty or representation for any use or purpose. Your
# use of it is subject to your agreement with Google.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================
echo "Loading environment variables from .env..."
source .env

# =============================================================================
# GOOGLE CLOUD SERVICES SETUP
# =============================================================================
echo "Setting up Google Cloud services and infrastructure..."

echo "Enabling required Google Cloud APIs..."
gcloud services enable container.googleapis.com            # Google Kubernetes Engine
gcloud services enable storage.googleapis.com              # Cloud Storage  
gcloud services enable tpu.googleapis.com                  # Tensor Processing Units
gcloud services enable compute.googleapis.com              # Compute Engine
gcloud services enable cloudbuild.googleapis.com           # Cloud Build
gcloud services enable serviceusage.googleapis.com         # Service Usage API
gcloud services enable iam.googleapis.com                  # Identity and Access Management

echo "Configuring service account permissions..."
# Grant Cloud Build service account storage admin permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${BUILD_SA}" \
  --role="roles/storage.admin"

# Grant compute service account necessary permissions for TPU and storage access
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/serviceusage.serviceUsageConsumer"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/storage.objectViewer"

echo "Creating Google Cloud Storage buckets..."
# Create output bucket for model checkpoints and logs
gcloud storage buckets create gs://${OUTPUT_BUCKET} \
  --location=${REGION} \
  --uniform-bucket-level-access || echo "Output bucket already exists"

# Create dataset bucket for training data
gcloud storage buckets create gs://${DATASET_BUCKET} \
  --location=${REGION} \
  --uniform-bucket-level-access || echo "Dataset bucket already exists"

echo "Google Cloud services setup complete!"

# =============================================================================
# KUBERNETES TOOLS INSTALLATION
# =============================================================================
echo "Installing Kubernetes management tools..."

echo "Installing kjob (Kubernetes Job Management)..."
curl -Lo kubectl-kjob https://github.com/kubernetes-sigs/kjob/releases/download/${KJOB_VERSION}/kubectl-kjob-linux-amd64
chmod +x kubectl-kjob
mkdir -p ~/.local/bin && mv kubectl-kjob ~/.local/bin/

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Apply Custom Resource Definitions
~/.local/bin/kubectl-kjob printcrds | kubectl apply --server-side -f -

echo "Installing kueue (Job Queueing System)..."
curl -Lo kubectl-kueue https://github.com/kubernetes-sigs/kueue/releases/download/${KUEUE_VERSION}/kubectl-kueue-linux-amd64
chmod +x kubectl-kueue
mkdir -p ~/.local/bin && mv kubectl-kueue ~/.local/bin/

# =============================================================================
# LOCAL ENVIRONMENT AND DOCKER SETUP
# =============================================================================
echo "Setting up local environment and building Docker images..."

# Install Python dependencies
uv sync --locked 

# Copy Google Cloud credentials for container use
cp ~/.config/gcloud/application_default_credentials.json .  

# Build and push the base Saturn image for training
echo "Building and pushing base training image: ${BASE_IMAGE}"
docker build -t ${BASE_IMAGE} .  
docker push ${BASE_IMAGE}

# =============================================================================
# XPK CLUSTER CREATION
# =============================================================================
echo "Creating XPK cluster with TPU resources..."

uv run xpk cluster create \
  --cluster ${CLUSTER_NAME} \
  --tpu-type=${TPU_TYPE} \
  --num-slices=${NUM_SLICES} \
  --zone ${ZONE} --on-demand \
  --custom-cluster-arguments='--enable-shielded-nodes --shielded-secure-boot --shielded-integrity-monitoring' \
  --custom-nodepool-arguments='--shielded-secure-boot --shielded-integrity-monitoring'

# =============================================================================
# STEP 1: MODEL CONVERSION (HUGGINGFACE → MAXTEXT)
# =============================================================================
echo "Step 1: Converting HuggingFace model to MaxText format..."
echo "Note: This requires significant memory resources and runs on TPU"

uv run xpk workload create \
  --workload make-model-gemma12b \
  --docker-image ${BASE_IMAGE} \
  --cluster ${CLUSTER_NAME} \
  --tpu-type ${TPU_TYPE} \
  --zone ${ZONE} \
  --num-slices=${NUM_SLICES} \
  --command "python3 -m MaxText.utils.ckpt_conversion.to_maxtext MaxText/configs/base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${MODEL_BUCKET}/${MODEL_VARIATION}/unscanned/${idx} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=${SCAN_LAYERS}" 

echo "Following conversion logs..."
kubectl logs $(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep make-model-gemma12b | head -n 1) -f

# =============================================================================
# STEP 2: MODEL VALIDATION TEST
# =============================================================================
echo "Step 2: Testing converted model with sample inference..."

uv run xpk workload create \
  --workload run-gemma12b-test \
  --docker-image ${BASE_IMAGE} \
  --cluster ${CLUSTER_NAME} \
  --tpu-type ${TPU_TYPE} \
  --zone ${ZONE} \
  --num-slices=${NUM_SLICES} \
  --command "python3 -m MaxText.decode MaxText/configs/base.yml \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    per_device_batch_size=1 \
    run_name=ht_test \
    max_prefill_predict_length=122 \
    max_target_length=300 \
    steps=1 \
    async_checkpointing=false \
    scan_layers=${SCAN_LAYERS} \
    use_multimodal=${USE_MULTIMODAL} \
    prompt='Describe image \<start_of_image\>' \
    image_path='MaxText/test_assets/test_image.jpg' \
    attention='dot_product'"

echo "Following test inference logs..."
kubectl logs $(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep run-gemma12b-test) -f

# =============================================================================
# STEP 3: SUPERVISED FINE-TUNING ON CHARTQA
# =============================================================================
echo "Step 3: Fine-tuning model on ChartQA dataset using SFT..."

uv run xpk workload create \
  --workload finetune-sft-gemma12b-chartqa \
  --docker-image ${BASE_IMAGE} \
  --cluster ${CLUSTER_NAME} \
  --tpu-type ${TPU_TYPE} \
  --zone ${ZONE} \
  --num-slices=${NUM_SLICES} \
  --env GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT} \
  --command "python -m MaxText.sft_trainer MaxText/configs/sft-vision-chartqa.yml \
    run_name=${idx} \
    model_name=${MODEL_NAME} \
    tokenizer_path='google/gemma-3-12b-it' \
    per_device_batch_size=8 \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    steps=${SFT_STEPS} \
    scan_layers=${SCAN_LAYERS} \
    async_checkpointing=False \
    attention=dot_product \
    dataset_type=hf \
    hf_path=HuggingFaceM4/ChartQA \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${BASE_OUTPUT_DIRECTORY} \
    load_parameters_path=${UNSCANNED_CKPT_PATH} \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    sharding_tolerance=0.05"

echo "Training started. You can monitor with TensorBoard:"
echo "tensorboard --logdir=${BASE_OUTPUT_DIRECTORY}/${idx}/tensorboard/"

echo "Following fine-tuning logs..."
kubectl logs $(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep finetune-sft-gemma12b-chartqa) -f

# =============================================================================
# STEP 4: INFERENCE TEST WITH FINE-TUNED MODEL
# =============================================================================
echo "Step 4: Testing fine-tuned model with inference..."

uv run xpk workload create \
  --workload infer-gemma3-12b-chartqa \
  --docker-image ${BASE_IMAGE} \
  --cluster ${CLUSTER_NAME} \
  --tpu-type ${TPU_TYPE} \
  --zone ${ZONE} \
  --num-slices=${NUM_SLICES} \
  --env GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT} \
  --command "python3 -m MaxText.decode MaxText/configs/base.yml \
    model_name=${MODEL_NAME} \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${FINETUNED_CKPT_PATH} \
    per_device_batch_size=1 \
    run_name=ht_test \
    max_prefill_predict_length=1024 \
    max_target_length=2048 \
    steps=1 \
    async_checkpointing=false \
    scan_layers=${SCAN_LAYERS} \
    use_multimodal=${USE_MULTIMODAL} \
    prompt='Describe image \<start_of_image\>' \
    image_path='MaxText/test_assets/test_image.jpg' \
    attention='dot_product'" 

echo "Following inference test logs..."
kubectl logs $(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep infer-gemma3-12b-chartqa) -f

# Expected response example:
# "The image shows a panoramic view of Seattle, Washington. The iconic Space Needle is prominently 
# featured in the center of the skyline, towering over the city. The city is densely populated with 
# modern skyscrapers and buildings, creating a bustling urban landscape. In the background, snow-capped 
# mountains are visible, adding a scenic backdrop to the city view. The sky is clear and blue with 
# scattered clouds.<end_of_turn>"

# =============================================================================
# STEP 5: BUILD AND DEPLOY JETSTREAM SERVING CONTAINER
# =============================================================================
echo "Step 5: Building JetStream serving container..."

cd inference 
echo "Building and pushing JetStream serving image: ${PLUTO_IMAGE}"
docker build -t ${PLUTO_IMAGE} .
docker push ${PLUTO_IMAGE}
cd ..

echo "Deploying model serving workload with JetStream..."
uv run xpk workload create \
  --workload serve-gemma3-12b-chartqa \
  --docker-image ${PLUTO_IMAGE} \
  --cluster ${CLUSTER_NAME} \
  --tpu-type ${TPU_TYPE} \
  --zone ${ZONE} \
  --num-slices=${NUM_SLICES} \
  --env GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT} \
  --command "python3 -m MaxText.maxengine_server \
    MaxText/configs/base.yml \
    tokenizer_path=${TOKENIZER_PATH} \
    load_parameters_path=${FINETUNED_CKPT_PATH} \
    max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
    max_target_length=${MAX_TARGET_LENGTH} \
    model_name=${MODEL_NAME} \
    ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
    ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
    ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
    scan_layers=${SCAN_LAYERS} \
    weight_dtype=${WEIGHT_DTYPE} \
    per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"

echo "Following serving deployment logs..."
kubectl logs $(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep serve-gemma3-12b-chartqa) -f

echo "Checking serving pod status..."
kubectl get pods -A | grep serve-gemma3-12b-chartqa

# =============================================================================
# INTERACTIVE SERVING AND TESTING SECTION
# =============================================================================
echo ""
echo "=== INTERACTIVE SERVING SETUP ==="
echo ""
echo "To test the deployed model, run these commands in separate terminals:"
echo ""
echo "1. Port forward to access the serving endpoint:"
echo "   kubectl port-forward \$(kubectl get pods --no-headers -o custom-columns=':metadata.name' | grep serve-gemma3-12b-chartqa) 9000:9000"
echo ""
echo "2. Run the JetStream container locally for testing:"
echo "   docker run --network host -p 9000:9000 -it ${PLUTO_IMAGE}"
echo ""
echo "3. Inside the container, run a warmup request:"
echo "   python /workspace/JetStream/jetstream/tools/requester.py --tokenizer assets/tokenizer.gemma3 --text 'The name of the tallest mountain in the world is ..'"
echo ""
echo "4. Run load test (512 queries):"
echo "   python /workspace/JetStream/jetstream/tools/load_tester.py --text 'The name of the tallest mountain in the world is ..'"
echo "   # Expected performance: ~22.89 QPS, ~22.37 seconds for 512 queries"
echo ""

# Pause for user interaction
read -p "Press Enter to continue with model export after testing serving..."

# Clean up serving workload
echo "Cleaning up serving workload..."
uv run xpk workload delete --workload serve-gemma3-12b-chartqa --cluster ${CLUSTER_NAME} --zone ${ZONE}

# =============================================================================
# STEP 6: EXPORT TO HUGGINGFACE FORMAT
# =============================================================================
echo "Step 6: Converting fine-tuned model back to HuggingFace format..."

uv run xpk workload create \
  --workload export-gemma3-12b-chartqa \
  --docker-image ${BASE_IMAGE} \
  --cluster ${CLUSTER_NAME} \
  --tpu-type ${TPU_TYPE} \
  --zone ${ZONE} \
  --num-slices=${NUM_SLICES} \
  --env GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT} \
  --command "python3 -m MaxText.utils.ckpt_conversion.to_huggingface MaxText/configs/base.yml \
    model_name=${MODEL_NAME} \
    hf_access_token=${HF_TOKEN} \
    load_parameters_path=${FINETUNED_CKPT_PATH} \
    base_output_directory=${LOCAL_PATH} \
    use_multimodal=${USE_MULTIMODAL} \
    scan_layers=${SCAN_LAYERS}"

echo "Following export logs..."
kubectl get pods | grep export-gemma3-12b-chartqa
kubectl logs $(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep export-gemma3-12b-chartqa) -f

# =============================================================================
# CLEANUP
# =============================================================================
echo ""
echo "=== EXPERIMENT COMPLETE ==="
echo ""
echo "Fine-tuned model exported to: ${LOCAL_PATH}"
echo ""
read -p "Do you want to delete the XPK cluster to avoid charges? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting XPK cluster..."
    yes | uv run xpk cluster delete --cluster ${CLUSTER_NAME} --zone ${ZONE}
    echo "Cluster deleted successfully!"
else
    echo "Cluster preserved. Remember to delete it manually to avoid charges:"
    echo "  uv run xpk cluster delete --cluster ${CLUSTER_NAME} --zone ${ZONE}"
fi

echo ""
echo "🎉 Gemma 3-12B fine-tuning experiment completed successfully!"
echo "✅ Model converted to MaxText format"
echo "✅ Fine-tuned on ChartQA dataset (${SFT_STEPS} steps)"
echo "✅ Deployed with JetStream for high-performance serving"
echo "✅ Exported back to HuggingFace format"
echo ""
echo "Results available in: ${LOCAL_PATH}"
