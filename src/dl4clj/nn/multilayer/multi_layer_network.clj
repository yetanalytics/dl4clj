(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/multilayer/MultiLayerNetwork.html"}
    dl4clj.nn.multilayer.multi-layer-network
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.nn.conf.constants :as enum]
            [dl4clj.nn.api.model :refer [fit!]]
            [dl4clj.helpers :refer [new-lazy-iter reset-if-empty?! reset-iterator!]]
            [dl4clj.datasets.api.iterators :refer [has-next? next-example!]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]])
  (:import [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder]))

(defn new-multi-layer-network
  "constructor for a multi-layer-network given a config and optionaly
  some params (INDArray or vec)"
  [& {:keys [conf params]}]
  (if params
    (MultiLayerNetwork. conf (vec-or-matrix->indarray params))
    (MultiLayerNetwork. conf)))

(defn initialize!
  "Sets the input and labels from this dataset

  :ds (dataset), a dataset
   -see: nd4clj.linalg.dataset.data-set
         dl4clj.datasets.datavec"
  [& {:keys [mln ds]}]
  (doto mln
    (.initialize ds)))

(defn initialize-layers!
  "initialize the neuralNets based on the input.

  :input (INDArray or vec), the input matrix for training"
  [& {:keys [mln input]}]
  (doto mln
    (.initializeLayers (vec-or-matrix->indarray input))))

(defn summary
  "String detailing the architecture of the multi-layer-network. (mln)"
  [mln]
  (.summary mln))

(defn train-mln-with-ds-iter!
  "train the supplied multi layer network on the supplied dataset

  :iter (iterator), an iterator wrapping a dataset

  :n-epochs (int), the number of passes through the dataset"
  [& {:keys [mln iter n-epochs]}]
  (dotimes [n n-epochs]
    (while (has-next? iter)
      ;; fit handles the iter resetting
      (fit! :mln mln :iter iter)))
  mln)

(defn train-mln-with-lazy-seq!
  "train the supplied multi layer network on the dataset contained within
   the supplied lazy seq

   :lazy-seq-data (lazy-seq), a lazy-seq of dataset objects
    - created by data-from-iter in: dl4clj.helpers"

  ;; test this
  [& {:keys [lazy-seq-data mln n-epochs]}]
  (dotimes [n n-epochs]
    (loop [_ mln
           accum! lazy-seq-data]
      ;; this could never complete
      ;; rest always returns a seq
      (if (not (empty? accum!))
        (let [data (first accum!)]
          (recur (fit! :mln mln :data data)
                 (rest accum!)))
        mln)))

  #_(dotimes [n n-epochs]
    ;; look into avoiding creation of lazy iter and just recursively going through
    ;; lazy seq

    ;; dont know of a more effecient way of doing this
    ;; you cant reset a lazy-seq-iter so i just make a new one
    ;; prob why training takes non-neglegible amount of time



    #_(let [iter (new-lazy-iter lazy-seq-data)]
      (while (has-next? iter)
        (let [nxt (next-example! iter)]
          (fit! :mln mln :data nxt))))))

(defn pre-train!
  "Perform layerwise pretraining on all pre-trainable layers in the network (VAEs, RBMs, Autoencoders, etc)
  Note that pretraining will be performed on one layer after the other, resetting the DataSetIterator between iterations.
  For multiple epochs per layer, appropriately wrap the iterator (for example, a MultipleEpochsIterator)
  or train each layer manually using (pre-train-layer! layer-idx DataSetIterator)

  :iter (ds-iter), dataset iterator
   - see: dl4clj.datasets.datavec"
  [& {:keys [mln iter]}]
  (.pretrain mln (reset-iterator! iter)))

(defn pre-train-layer!
  "Perform layerwise unsupervised training on a single pre-trainable layer
  in the network (VAEs, RBMs, Autoencoders, etc) If the specified layer index
  (0 to n-layers - 1) is not a pretrainable layer, this is a no-op.

  :layer-idx (int), the index of the layer you want to pretrain

  :iter (ds-iter), dataset iterator
   - see: dl4clj.datasets.datavec

  :features (INDArray or vec), training data array"
  [& {:keys [mln layer-idx iter features]
      :as opts}]
  (cond (contains-many? opts :layer-idx :iter)
        (.pretrainLayer mln layer-idx (reset-iterator! iter))
        (contains-many? opts :layer-idx :features)
        (.pretrainLayer mln layer-idx (vec-or-matrix->indarray features))
        :else
        (assert false "you must supply the layer's index and either a dataset
 iterator or an array of features to pretrain on")))

(defn fine-tune!
  "Run SGD based on the given labels

  returns the fine tuned model"
  [mln]
  (doto mln
    (.finetune)))

(defn evaluate-classification
  ;; might be a duplicate fn
  "if you only supply mln and iter: Evaluate the network (classification performance)
  if you supply mln, iter and labels-list: Evaluate the network on the provided data set.
  if you supply all args: Evaluate the network (for classification) on the provided data set,
                          with top N accuracy in addition to standard accuracy.

  :iter (ds-iter), a dataset iterator
   - see: dl4clj.datasets.datavec

  :labels (coll), a collection of strings (the labels)

  :top-n (int), N value for top N accuracy evaluation"
  [& {:keys [mln iter labels top-n]
      :as opts}]
  (let [ds-iter (reset-iterator! iter)]
    (cond (contains-many? opts :labels-list :top-n)
          (.evaluate mln ds-iter (into '() labels) top-n)
          (contains? opts :labels-list)
          (.evaluate mln ds-iter (into '() labels))
          :else
          (.evaluate mln ds-iter))))

(defn evaluate-regression
  "Evaluate the network for regression performance

  :iter (ds-iter), a dataset iterator
   - see: dl4clj.datasets.datavec"
  ;; update doc
  [& {:keys [mln iter]}]
  (.evaluateRegression mln (reset-iterator! iter)))

(defn evaluate-roc
  "Evaluate the network (must be a binary classifier) on the specified data
   - see:dl4clj.eval.roc.rocs

  :iter (ds-iter), a dataset iterator
   - see: dl4clj.datasets.datavec

  :roc-threshold-steps (int), value needed to call the ROC constructor
   - see: dl4clj.eval.roc.rocs"
  [& {:keys [mln iter roc-threshold-steps]}]
  (.evaluateROC mln (reset-iterator! iter) roc-threshold-steps))

(defn evaluate-roc-multi-class
  "Evaluate the network on the specified data.

  :iter (ds-iter), a dataset iterator
   - see: dl4clj.datasets.datavec

  :roc-threshold-steps (int), value needed to call the ROCMultiClass constructor
   - see: dl4clj.eval.roc.rocs"
  [& {:keys [mln iter roc-threshold-steps]}]
  (.evaluateROCMultiClass mln (reset-iterator! iter) roc-threshold-steps))

(defn score-examples
  "Calculate the score for each example in a DataSet individually.
   - this fn allows for examples to be scored individually (at test time only),
     which may be useful for example for autoencoder architectures and the like.

  :dataset (datatset),
   -see: nd4clj.linalg.dataset.data-set
         dl4clj.datasets.datavec

  :add-regularization-terms? (boolean), if true, add l1/l2 terms to the score
   otherwise just return the scores

  :iter (ds-iter), dataset iterator
   - see: dl4clj.datasets.datavec"
  [& {:keys [mln dataset add-regularization-terms? iter]
      :as opts}]
  (cond (contains-many? opts :dataset :add-regularization-terms?)
        (.scoreExamples mln dataset add-regularization-terms?)
        (contains-many? opts :iter :add-regularization-terms?)
        (.scoreExamples mln (reset-iterator! iter) add-regularization-terms?)
        :else
        (assert false "you must supply data in the form of a dataset or a dataset iterator.
you must also supply whether or not you want to add regularization terms (L1, L2, dropout...)")))

(defn output
  "label the probabilities of the input or if masks are supplied,
  calculate the output of the network with masking arrays

  :iter (ds-iter), a dataset iterator
   - see: dl4clj.datasets.datavec

  :train? (boolean), are we in training mode?
   - This mainly affect hyper parameters such as drop out
     where certain things should be applied with activations

  :input (INDArray or vec), the input to label

  :features-mask (INDArray or vec), the mask used for the features

  :labels-mask (INDArray or vec), the mask used for the labels

  :training-mode (keyword), another way to say if its training or testing mode

  NOTE: this fn only resets the iterator if its empty"
  [& {:keys [iter train? input features-mask labels-mask
             training-mode mln]
      :as opts}]
  (let [ds-iter (if (contains? opts :iter)
                  (reset-if-empty?! iter))
        i (if (contains? opts :input)
            (vec-or-matrix->indarray input))]
    (cond (contains-many? opts :input :train?
                          :features-mask :labels-mask)
          (.output mln i train?
                   (vec-or-matrix->indarray features-mask)
                   (vec-or-matrix->indarray labels-mask))
          (contains-many? opts :training-mode :input)
          (.output mln i (enum/value-of {:layer-training-mode training-mode}))
          (contains-many? opts :train? :input)
          (.output mln i train?)
          (contains-many? opts :iter :train?)
          (.output mln ds-iter train?)
          (contains? opts :input)
          (.output mln i)
          (contains? opts :iter)
          (.output mln ds-iter)
          :else
          (assert false "you must supply atleast an input or iterator"))))



(defn reconstruct
  "reconstructs the input from the output of a given layer

  :layer-output (INDArray or vec), the input to transform

  :layer-idx (int), the layer to output for encoding

  returns a reconstructed matrix relative to the size of the last hidden layer
   - normally a probability distribution summing to one"
  [& {:keys [mln layer-output layer-idx]
      :as opts}]
  (assert (contains-many? opts :layer-output :layer-idx) "you must supply a layer and the input")
  (.reconstruct mln (vec-or-matrix->indarray layer-output) layer-idx))


(defn rnn-time-step
  "If this MultiLayerNetwork contains one or more RNN layers:
  conduct forward pass (prediction) but using previous stored state for any RNN layers.
   -  The activations for the final step are also stored in the RNN layers for
      use next time this fn is called.

  :input (INDArray or vec), Input to network. May be for one or multiple time steps.
   - For single time step: input has shape [miniBatchSize,inputSize] or [miniBatchSize,inputSize,1].
       - miniBatchSize=1 for single example.
   - For multiple time steps: [miniBatchSize,inputSize,inputTimeSeriesLength]"
  [& {:keys [mln input]}]
  (.rnnTimeStep mln (vec-or-matrix->indarray input)))
