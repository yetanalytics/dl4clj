(ns dl4clj.nn.api.multi-layer-network
  (:import [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.api Layer])
  (:require [dl4clj.utils :refer [contains-many? array-of obj-or-code? eval-if-code]]
            [dl4clj.helpers :refer [new-lazy-iter reset-if-empty?! reset-iterator!]]
            [clojure.core.match :refer [match]]
            [dl4clj.constants :as enum]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn initialize!
  "Sets the input and labels from this dataset

  :ds (dataset), a dataset
   -see: nd4clj.linalg.dataset.data-set"
  [& {:keys [mln ds as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :ds (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~mln (.initialize ~ds)))
         :else
         (let [[model d] (eval-if-code [mln seq?] [ds seq?])]
           (doto model (.initialize d)))))

(defn evaluate-classification
  "if you only supply mln and iter: Evaluate the network (classification performance)
  if you supply mln, iter and labels-list: Evaluate the network on the provided data set.
  if you supply all args: Evaluate the network (for classification) on the provided data set,
                          with top N accuracy in addition to standard accuracy.

  :iter (ds-iter), a dataset iterator
   - see: dl4clj.datasets.iterators

  :labels (coll), a collection of strings (the labels)

  :top-n (int), N value for top N accuracy evaluation"
  [& {:keys [mln iter labels top-n as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :iter (_ :guard seq?)
           :labels-list (:or (_ :guard vector?)
                             (_ :guard seq?))
           :top-n (:or (_ :guard number?)
                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.evaluate ~mln ~iter (into '() ~labels) (int ~top-n)))
         [{:mln _ :iter _ :labels-list _ :top-n _}]
         (let [[model i ls t-n] (eval-if-code [mln seq?]
                                              [iter seq?]
                                              [labels seq? vector?]
                                              [top-n seq? number?])]
           (.evaluate model (reset-iterator! i) (reverse (into '() ls)) t-n))
         [{:mln (_ :guard seq?)
           :iter (_ :guard seq?)
           :labels-list (:or (_ :guard vector?)
                             (_ :guard seq?))}]
         (obj-or-code? as-code? `(.evaluate ~mln ~iter (reverse (into '() ~labels))))
         [{:mln _ :iter _ :labels-list _}]
         (let [[model i ls] (eval-if-code [mln seq?] [iter seq?]
                                          [labels seq? vector?])]
          (.evaluate model (reset-iterator! i) (reverse (into '() ls))))
         [{:mln (_ :guard seq?)
           :iter (_ :guard seq?)}]
         (obj-or-code? as-code? `(.evaluate ~mln ~iter))
         :else
         (let [[model i] (eval-if-code [mln seq?]
                                       [iter seq?])]
           (.evaluate model (reset-iterator! i)))))

(defn evaluate-regression
  "Evaluate the network for regression performance

  :iter (ds-iter), a dataset iterator
   - see: dl4clj.datasets.iterators"
  [& {:keys [mln iter as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :iter (_ :guard seq?)}]
         (obj-or-code? as-code? `(.evaluateRegression ~mln ~iter))
         :else
         (let [[model i] (eval-if-code [mln seq?] [iter seq?])]
           (.evaluateRegression model (reset-iterator! i)))))

(defn evaluate-roc
  "Evaluate the network (must be a binary classifier) on the specified data
   - see:dl4clj.eval.roc.rocs

  :iter (ds-iter), a dataset iterator
   - see: dl4clj.datasets.iterators

  :roc-threshold-steps (int), value needed to call the ROC constructor
   - see: dl4clj.eval.roc.rocs"
  [& {:keys [mln iter roc-threshold-steps as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :iter (_ :guard seq?)
           :roc-threshold-steps (:or (_ :guard number?)
                                     (_ :guard seq?))}]
         (obj-or-code? as-code? `(.evaluateROC ~mln ~iter (int ~roc-threshold-steps)))
         :else
         (let [[model i steps] (eval-if-code [mln seq?] [iter seq?]
                                             [roc-threshold-steps seq? number?])]
           (.evaluateROC model (reset-iterator! i) steps))))

(defn evaluate-roc-multi-class
  "Evaluate the network on the specified data.

  :iter (ds-iter), a dataset iterator
   - see: dl4clj.datasets.iterators

  :roc-threshold-steps (int), value needed to call the ROCMultiClass constructor
   - see: dl4clj.eval.roc.rocs"
  [& {:keys [mln iter roc-threshold-steps as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :iter (_ :guard seq?)
           :roc-threshold-steps (:or (_ :guard number?)
                                     (_ :guard seq?))}]
         (obj-or-code? as-code? `(.evaluateROCMultiClass ~mln ~iter (int ~roc-threshold-steps)))
         :else
         (let [[model i steps] (eval-if-code [mln seq?] [iter seq?]
                                             [roc-threshold-steps seq? number?])]
           (.evaluateROCMultiClass model (reset-iterator! i) steps))))

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
   - see: dl4clj.datasets.iterators"
  [& {:keys [mln dataset add-regularization-terms? iter as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :dataset (_ :guard seq?)
           :add-regularization-terms? (:or (_ :guard boolean?)
                                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.scoreExamples ~mln ~dataset ~add-regularization-terms?))
         [{:mln _ :dataset _ :add-regularization-terms? _}]
         (let [[model ds terms?] (eval-if-code [mln seq?] [dataset seq?]
                                               [add-regularization-terms? seq? boolean?])]
           (.scoreExamples model ds terms?))
         [{:mln (_ :guard seq?)
           :iter (_ :guard seq?)
           :add-regularization-terms? (:or (_ :guard boolean?)
                                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.scoreExamples ~mln ~iter ~add-regularization-terms?))
         [{:mln _ :iter _ :add-regularization-terms? _}]
         (let [[model i terms?] (eval-if-code [mln seq?] [iter seq?]
                                              [add-regularization-terms? seq? boolean?])]
           (.scoreExamples model (reset-iterator! i) terms?))))

(defn output
  "label the probabilities of the input or if masks are supplied,
  calculate the output of the network with masking arrays

  :iter (ds-iter), a dataset iterator
   - see: dl4clj.datasets.iterators

  :train? (boolean), are we in training mode?
   - This mainly affect hyper parameters such as drop out
     where certain things should be applied with activations

  :input (INDArray or vec), the input to label

  :features-mask (INDArray or vec), the mask used for the features

  :labels-mask (INDArray or vec), the mask used for the labels

  :training-mode (keyword), another way to say if its training or testing mode

  NOTE: this fn only resets the iterator if its empty"
  [& {:keys [iter train? input features-mask labels-mask
             training-mode mln as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :train? (:or (_ :guard boolean?)
                        (_ :guard seq?))
           :features-mask (:or (_ :guard vector?)
                               (_ :guard seq?))
           :labels-mask (:or (_ :guard vector?)
                             (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.output ~mln (vec-or-matrix->indarray ~input) ~train?
                    (vec-or-matrix->indarray ~features-mask)
                    (vec-or-matrix->indarray ~labels-mask)))
         [{:mln _ :input _ :train? _ :features-mask _ :labels-mask _}]
         (let [[model i t? f-mask l-mask] (eval-if-code [mln seq?]
                                                        [input seq?]
                                                        [train? seq? boolean?]
                                                        [features-mask seq?]
                                                        [labels-mask seq?])]
          (.output model (vec-or-matrix->indarray i) t?
                   (vec-or-matrix->indarray f-mask)
                   (vec-or-matrix->indarray l-mask)))
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :training-mode (:or (_ :guard keyword?)
                               (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.output ~mln (vec-or-matrix->indarray ~input)
                   (enum/value-of {:layer-training-mode ~training-mode})))
         [{:mln _ :input _ :training-mode _}]
         (let [[model i t-m] (eval-if-code [mln seq?] [input set?]
                                           [training-mode seq? keyword?])]
          (.output model (vec-or-matrix->indarray i)
                  (enum/value-of {:layer-training-mode t-m})))
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :train? (:or (_ :guard boolean?)
                        (_ :guard seq?))}]
         (obj-or-code? as-code? `(.output ~mln (vec-or-matrix->indarray ~input) ~train?))
         [{:mln _ :input _ :train? _}]
         (let [[model i t?] (eval-if-code [mln seq?] [input seq?]
                                          [train? seq? boolean?])]
           (.output model (vec-or-matrix->indarray i) t?))
         [{:mln (_ :guard seq?)
           :iter (_ :guard seq?)
           :train? (:or (_ :guard boolean?)
                        (_ :guard seq?))}]
         (obj-or-code? as-code? `(.output ~mln ~iter ~train?))
         [{:mln _ :iter _ :train? _}]
         (let [[model i t?] (eval-if-code [mln seq?] [iter seq?] [train? seq? boolean?])]
           (.output model (reset-if-empty?! i) t?))
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.output ~mln (vec-or-matrix->indarray ~input)))
         [{:mln _ :input _}]
         (let [[model i] (eval-if-code [mln seq?] [input seq?])]
           (.output model (vec-or-matrix->indarray i)))
         [{:mln (_ :guard seq?)
           :iter (_ :guard seq?)}]
         (obj-or-code? as-code? `(.output ~mln ~iter))
         [{:mln _ :iter _}]
         (let [[model i] (eval-if-code [mln seq?] [iter seq?])]
           (.output model (reset-if-empty?! i)))))


(defn initialize-layers!
  "initialize the neuralNets based on the input.

  :input (INDArray or vec), the input matrix for training"
  [& {:keys [mln input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~mln
            (.initializeLayers (vec-or-matrix->indarray ~input))))
         :else
         (let [[model i] (eval-if-code [mln seq?] [input seq?])]
          (doto model (.initializeLayers (vec-or-matrix->indarray i))))))

(defn pre-train!
  "Perform layerwise pretraining on all pre-trainable layers in the network (VAEs, RBMs, Autoencoders, etc)
  Note that pretraining will be performed on one layer after the other, resetting the DataSetIterator between iterations.
  For multiple epochs per layer, appropriately wrap the iterator (for example, a MultipleEpochsIterator)
  or train each layer manually using (pre-train-layer! layer-idx DataSetIterator)

  :iter (ds-iter), dataset iterator
   - see: dl4clj.datasets.iterators"
  [& {:keys [mln iter as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :iter (_ :guard seq?)}]
         (obj-or-code? as-code? `(.pretrain ~mln ~iter))
         :else
         (let [[model i] (eval-if-code [mln seq?] [iter seq?])]
           (.pretrain model (reset-iterator! i)))))

(defn pre-train-layer!
  "Perform layerwise unsupervised training on a single pre-trainable layer
  in the network (VAEs, RBMs, Autoencoders, etc) If the specified layer index
  (0 to n-layers - 1) is not a pretrainable layer, this is a no-op.

  :layer-idx (int), the index of the layer you want to pretrain

  :iter (ds-iter), dataset iterator
   - see: dl4clj.datasets.iterators

  :features (INDArray or vec), training data array"
  [& {:keys [mln layer-idx iter features as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))
           :iter (_ :guard seq?)}]
         (obj-or-code? as-code? `(.pretrainLayer ~mln (int ~layer-idx) ~iter))
         [{:mln _ :layer-idx _ :iter _}]
         (let [[model l-idx i] (eval-if-code [mln seq?] [layer-idx seq? number?]
                                             [iter seq?])]
           (.pretrainLayer model l-idx (reset-iterator! i)))
         [{:mln (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.pretrainLayer ~mln (int ~layer-idx) (vec-or-matrix->indarray ~features)))
         [{:mln _ :layer-idx _ :features _}]
         (let [[model l-idx f] (eval-if-code [mln seq?] [layer-idx seq? number?]
                                             [features seq?])]
           (.pretrainLayer model l-idx (vec-or-matrix->indarray f)))))

(defn fine-tune!
  "Run SGD based on the given labels

  returns the fine tuned model"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~mln .finetune))
         :else
         (doto mln .finetune)))

(defn rnn-time-step
  "If this MultiLayerNetwork contains one or more RNN layers:
  conduct forward pass (prediction) but using previous stored state for any RNN layers.
   -  The activations for the final step are also stored in the RNN layers for
      use next time this fn is called.

  :input (INDArray or vec), Input to network. May be for one or multiple time steps.
   - For single time step: input has shape [miniBatchSize,inputSize] or [miniBatchSize,inputSize,1].
       - miniBatchSize=1 for single example.
   - For multiple time steps: [miniBatchSize,inputSize,inputTimeSeriesLength]"
  [& {:keys [mln input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.rnnTimeStep ~mln (vec-or-matrix->indarray ~input)))
         :else
         (let [[model i] (eval-if-code [mln seq?] [input seq?])]
           (.rnnTimeStep model (vec-or-matrix->indarray i)))))

(defn reconstruct
  "reconstructs the input from the output of a given layer

  :layer-output (INDArray or vec), the input to transform

  :layer-idx (int), the layer to output for encoding

  returns a reconstructed matrix relative to the size of the last hidden layer
   - normally a probability distribution summing to one"
  [& {:keys [mln layer-output layer-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :layer-output (:or (_ :guard vector?)
                              (_ :guard seq?))
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.reconstruct ~mln (vec-or-matrix->indarray ~layer-output) (int ~layer-idx)))
         :else
         (let [[model l-o l-idx] (eval-if-code [mln seq?] [layer-output seq?]
                                               [layer-idx seq? number?])]
           (.reconstruct model (vec-or-matrix->indarray l-o) l-idx))))

(defn summary
  "String detailing the architecture of the multi-layer-network. (mln)"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.summary ~mln))
         :else
         (.summary mln)))

(defn activate-selected-layers
  "Calculate activation for few layers at once. Suitable for autoencoder partial activation

  returns the activation from the last layer

   :from (int), starting layer idx

   :to (int), ending layer idx

   :input (INDArray or vec), the input to propagate through the layers"
  [& {:keys [mln from to input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :from (:or (_ :guard number?)
                      (_ :guard seq?))
           :to (:or (_ :guard number?)
                    (_ :guard seq?))
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.activateSelectedLayers ~mln (int ~from)
                                   (int ~to)
                                   (vec-or-matrix->indarray ~input)))
         :else
         (let [[model f t i] (eval-if-code [mln seq?] [from seq? number?]
                                           [to seq? number?] [input seq?])]
           (.activateSelectedLayers model f t (vec-or-matrix->indarray i)))))

(defn activate-from-prev-layer
  "Calculate activation from previous layer including pre processing where necessary

  :current-layer-idx (int), the index of the current layer
   - you will get the activation from the layer directly before this one

  :input (INDArray or vec), the input to propagate through the layers

  :training? (boolean), is this training mode?"
  [& {:keys [mln current-layer-idx input training? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :current-layer-idx (:or (_ :guard number?)
                                   (_ :guard seq?))
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.activationFromPrevLayer ~mln ~current-layer-idx
                                    (vec-or-matrix->indarray ~input) ~training?))
         :else
         (let [[model cur-idx i t?] (eval-if-code [mln seq?]
                                                  [current-layer-idx seq? number?]
                                                  [input seq?]
                                                  [training? seq? boolean?])]
           (.activationFromPrevLayer model cur-idx (vec-or-matrix->indarray i) t?))))

(defn clear-layer-mask-arrays!
  "Remove the mask arrays from all layers.

  returns the multi layer network after the mutation"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~mln .clearLayerMaskArrays))
         :else
         (doto mln .clearLayerMaskArrays)))

(defn compute-z
  "if you only supply training?: Compute input linear transformation (z) of the output layer
  if you supply training? and input: Compute activations from input to output of the output layer
   - both ways return the list of activations for each layer

  :training? (boolean), training mode?

  :input (INDArray or vec), the input to propagate through the network for calcing activations"
  [& {:keys [mln training? input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.computeZ ~mln (vec-or-matrix->indarray ~input) ~training?))
         [{:mln _ :training? _ :input _}]
         (let [[model i t?] (eval-if-code [mln seq?] [input seq?]
                                          [training? seq? boolean?])]
           (.computeZ model (vec-or-matrix->indarray i) t?))
         [{:mln (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.computeZ ~mln ~training?))
         :else
         (let [[model t?] (eval-if-code [mln seq?] [training? seq? boolean?])]
           (.computeZ model t?))))

(defn get-epsilon
  "returns epsilon for a given multi-layer-network (mln)"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.epsilon ~mln))
         :else
         (.epsilon mln)))

(defn feed-forward
  "if :features-mask and :labels-mask supplied:

   Compute the activations from the input to the output layer,
   given mask arrays (that may be null) The masking arrays are used in situations
   such an one-to-many and many-to-one rucerrent neural network (RNN) designs,
   as well as for supporting time series of varying lengths within the same minibatch for RNNs.

  else, just compute the activations from the input to the output layer

  :train? (boolean), is this training mode?

  :input (INDArray or vec), the input to be propagated through the network

  :features-mask (INDArray or vec), mask for the input features

  :labels-mask (INDArray or vec), mask for the labels"
  [& {:keys [mln train? input features-mask labels-mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :features-mask (:or (_ :guard vector?)
                               (_ :guard seq?))
           :labels-mask (:or (_ :guard vector?)
                             (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.feedForward ~mln (vec-or-matrix->indarray ~input)
                         (vec-or-matrix->indarray ~features-mask)
                         (vec-or-matrix->indarray ~labels-mask)))
         [{:mln _ :input _ :features-mask _ :labels-mask _}]
         (let [[model i f-m l-m] (eval-if-code [mln seq?]
                                               [input seq?]
                                               [features-mask seq?]
                                               [labels-mask seq?])]
           (.feedForward model (vec-or-matrix->indarray i)
                         (vec-or-matrix->indarray f-m)
                         (vec-or-matrix->indarray l-m)))
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :train? (:or (_ :guard boolean?)
                        (_ :guard seq?))}]
         (obj-or-code? as-code? `(.feedForward ~mln (vec-or-matrix->indarray ~input) ~train?))
         [{:mln _ :input _ :train? _}]
         (let [[model i t?] (eval-if-code [mln seq?] [input seq?]
                                          [train? seq? boolean?])]
           (.feedForward model (vec-or-matrix->indarray i) t?))
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.feedForward ~mln (vec-or-matrix->indarray ~input)))
         [{:mln _ :input _}]
         (let [[model i] (eval-if-code [mln seq?] [input seq?])]
           (.feedForward model (vec-or-matrix->indarray i)))
         [{:mln (_ :guard seq?)
           :train? (:or (_ :guard boolean?)
                        (_ :guard seq?))}]
         (obj-or-code? as-code? `(.feedForward ~mln ~train?))
         [{:mln _ :train? _}]
         (let [[model t?] (eval-if-code [mln seq?] [train? seq? boolean?])]
           (.feedForward model t?))
         [{:mln (_ :guard seq?)}]
         (obj-or-code? as-code? `(.feedForward ~mln))
         :else
         (.feedForward mln)))

(defn feed-forward-to-layer
  "Compute the activations from the input to the specified layer.
   - if input is not supplied, uses the currently set input for the mln

  :layer-idx (int), the index of the layer you want the input propagated through

  :train? (boolean), are we in training mode?

  :input (INDArray or vec), the input to propagate through the specified layer

  Note: the returned output list contains the original input at idx 0"
  [& {:keys [mln layer-idx train? input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))
           :train? (:or (_ :guard boolean?)
                        (_ :guard seq?))
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.feedForwardToLayer ~mln (int ~layer-idx)
                               (vec-or-matrix->indarray ~input)
                               ~train?))
         [{:mln _ :layer-idx _ :train? _ :input _}]
         (let [[model l-idx i t?] (eval-if-code [mln seq?] [layer-idx seq? number?]
                                                [input seq?] [train? seq? boolean?])]
           (.feedForwardToLayer model l-idx (vec-or-matrix->indarray i) t?))
         [{:mln (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.feedForwardToLayer ~mln (int ~layer-idx)
                               (vec-or-matrix->indarray ~input)))
         [{:mln _ :layer-idx _ :input _}]
         (let [[model l-idx i] (eval-if-code [mln seq?] [layer-idx seq? number?]
                                             [input seq?])]
           (.feedForwardToLayer model l-idx (vec-or-matrix->indarray i)))
         [{:mln (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))
           :train? (:or (_ :guard boolean?)
                        (_ :guard seq?))}]
         (obj-or-code? as-code? `(.feedForwardToLayer ~mln (int ~layer-idx) ~train?))
         [{:mln _ :layer-idx _ :train? _}]
         (let [[model l-idx t?] (eval-if-code [mln seq?] [layer-idx seq? number?]
                                              [train? seq? boolean?])]
           (.feedForwardToLayer model l-idx t?))))

(defn get-default-config
  "gets the default config for the multi-layer-network"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getDefaultConfiguration ~mln))
         :else
         (.getDefaultConfiguration mln)))

(defn get-input
  "return the input to the mln"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getInput ~mln))
         :else
         (.getInput mln)))

(defn get-layer
  "return the layer of the mln based on its position within the mln

  :layer-idx (int), the index of the layer you want to get from the mln

  :layer-name (str), the name of the layer you want to get from the mln"
  [& {:keys [mln layer-idx layer-name as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getLayer ~mln (int ~layer-idx)))
         [{:mln _ :layer-idx _}]
         (let [[model l-idx] (eval-if-code [mln seq?] [layer-idx seq? number?])]
           (.getLayer model l-idx))
         [{:mln (_ :guard seq?)
           :layer-name (:or (_ :guard string?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getLayer ~mln ~layer-name))
         [{:mln _ :layer-name _}]
         (let [[model l-name] (eval-if-code [mln seq?] [layer-name seq? string?])]
           (.getLayer model l-name))))

(defn get-layer-names
  "return a list of the layer names in the mln"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLayerNames ~mln))
         :else
         (.getLayerNames mln)))

(defn get-layers
  "returns an array of the layers within the mln"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLayers ~mln))
         :else
         (.getLayers mln)))

(defn get-layer-wise-config
  "returns the configuration for the layers in the mln"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLayerWiseConfigurations ~mln))
         :else
         (.getLayerWiseConfigurations mln)))

(defn get-mask
  "return the mask array used in this mln"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getMask ~mln))
         :else
         (.getMask mln)))

(defn get-n-layers
  "get the number of layers in the mln"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getnLayers ~mln))
         :else
         (.getnLayers mln)))

(defn get-output-layer
  "returns the output layer of the mln"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getOutputLayer ~mln))
         :else
         (.getOutputLayer mln)))

(defn get-updater
  "return the updater used in this mln"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getUpdater ~mln))
         :else
         (.getUpdater mln)))

(defn init-gradients-view!
  "initializes the flattened gradients array (used in backprop) and
  sets the appropriate subset in all layers.

  - this gets called behind the scene when using fit!"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~mln .initGradientsView))
         :else
         (doto mln .initGradientsView)))

(defn get-mln-input
  "returns the input/feature matrix for the model"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.input ~mln))
         :else
         (.input mln)))

(defn is-init-called?
  "was the model initialized"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.isInitCalled ~mln))
         :else
         (.isInitCalled mln)))

(defn print-config
  "Prints the configuration and returns the mln"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~mln .printConfiguration))
         :else
         (doto mln .printConfiguration)))

(defn rnn-activate-using-stroed-state
  "returns the activation of the rnn given its most recent state
   - does not modify the RNN layer state, pure fn

  :input (INDArray or vec), the input fed to the rnn

  :training? (boolean), is this training mode?

  :store-last-for-tbptt? (boolean), set to true if used as part of truncated bptt training

  returns the activations for each layer
   - the input is idx 0, followed by the activations"
  [& {:keys [mln input training? store-last-for-tbptt? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))
           :store-last-for-tbptt? (:or (_ :guard boolean?)
                                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.rnnActivateUsingStoredState ~mln (vec-or-matrix->indarray ~input)
                                        ~training? ~store-last-for-tbptt?))
         :else
         (let [[model i t? store-last?] (eval-if-code [mln seq?] [input seq?]
                                                      [training? seq? boolean?]
                                                      [store-last-for-tbptt? seq? boolean?])]
           (.rnnActivateUsingStoredState model (vec-or-matrix->indarray i)
                                         t? store-last?))))

(defn rnn-clear-prev-state!
  "clear the previous state of the rnn layers if any and return the mln"
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~mln .rnnClearPreviousState))
         :else
         (doto mln .rnnClearPreviousState)))

(defn rnn-get-prev-state
  "get the state of the rnn layer given its index in the mln

  :layer-idx (int), the index of the rnn within the mln"
  [& {:keys [mln layer-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.rnnGetPreviousState ~mln (int ~layer-idx)))
         :else
         (let [[model l-idx] (eval-if-code [mln seq?] [layer-idx seq? number?])]
           (.rnnGetPreviousState model l-idx))))

(defn rnn-set-prev-state!
  "Set the state of the RNN layer and return the updated mln

  :layer-idx (int), the index of the rnn within the mln

  :state (map), {str INDArray}, The state to set the specified layer to

  returns the mln"
  [& {:keys [mln layer-idx state as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))
           :state (:or (_ :guard map?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~mln
            (.rnnSetPreviousState (int ~layer-idx) ~state)))
         :else
         (let [[model l-idx s] (eval-if-code [mln seq?]
                                             [layer-idx seq? number?]
                                             [state seq? map?])]
          (doto model (.rnnSetPreviousState l-idx s)))))

(defn set-mln-input!
  "Note that if input isn't nil and the neuralNets are nil,
  this is a way of initializing the neural network, returns the mln

  :input (INDArray or vec), the input to the mln"
  [& {:keys [mln input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~mln
            (.setInput (vec-or-matrix->indarray ~input))))
         :else
         (let [[model i] (eval-if-code [mln seq?] [input seq?])]
          (doto model (.setInput (vec-or-matrix->indarray i))))))

(defn set-labels-mln!
  "sets the labels given an array of labels,
  returns the mln.

  :labels (INDArray or vec), the labels to be set"
  [& {:keys [mln labels as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~mln
            (.setLabels (vec-or-matrix->indarray ~labels))))
         :else
         (let [[model l] (eval-if-code [mln seq?] [labels seq?])]
          (doto model (.setLabels (vec-or-matrix->indarray l))))))

(defn set-layers!
  "sets the layers of the mln in the order in which they appear in the supplied coll.

  :layers (coll), a collection of layers to add to the mln

  returns the mln"
  [& {:keys [mln layers as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :layers _}]
         (obj-or-code?
          as-code?
          `(doto ~mln
           (.setLayers (array-of :data ~layers
                                 :java-type Layer))))
         :else
         (let [[model ls] (eval-if-code [mln seq?] [layers seq? coll?])]
          (doto model (.setLayers (array-of :data ls :java-type Layer))))))

(defn set-layer-wise-config!
  "sets the configuration for a mln given a multi-layer configuration.
  returns the mln

  :mln (multi layer network), the multi layer network

  :multi-layer-conf (multi layer conf), the configuration for the multi layer network

  NOTE: you should not need this fn.  You can set the multi-layer-conf when creating your mln
   - see: new-multi-layer-network at the top of this ns"
  [& {:keys [mln multi-layer-conf as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :multi-layer-conf (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(doto ~mln
            (.setLayerWiseConfigurations ~multi-layer-conf)))
         :else
         (let [[model conf] (eval-if-code [mln seq?] [multi-layer-conf seq?])]
          (doto model (.setLayerWiseConfigurations conf)))))

(defn set-mask!
  "set the mask, returns the mln

  :mask (INDArray or vec), the mask to set for the mln"
  [& {:keys [mln mask as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :mask (:or (_ :guard vector?)
                      (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~mln
            (.setMask (vec-or-matrix->indarray ~mask))))
         :else
         (let [[model m] (eval-if-code [mln seq?] [mask seq?])]
          (doto model (.setMask (vec-or-matrix->indarray m))))))

(defn set-parameters!
  "set the paramters for this model (mln).
   - This is used to manipulate the weights and biases across all neuralNets
     (including the output layer)

  :params (INDArray or vec), a parameter vector equal 1,numParameters

  returns the mln"
  [& {:keys [mln params as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :params (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~mln
            (.setParameters (vec-or-matrix->indarray ~params))))
         :else
         (let [[model p] (eval-if-code [mln seq?] [params seq?])]
          (doto model (.setParameters (vec-or-matrix->indarray p))))))

(defn update-mln!
  "Assigns the parameters of mln to the ones specified by another mln.
  This is used in loading from input streams, factory methods, etc
   - returns mln

  you can also use update! in the model interface ns"
  [& {:keys [mln other-mln as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :other-mln (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~mln (.update ~other-mln)))
         :else
         (let [[model1 model2] (eval-if-code [mln seq?] [other-mln seq?])]
           (doto model1 (.update model2)))))

(defn update-rnn-state-with-tbptt-state!
  "updates the rnn state to be that of the tbptt state.
  returns the mln."
  [mln & {:keys [as-code?]
          :or {as-code? true}}]
  (match [mln]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~mln .updateRnnStateWithTBPTTState))
         :else
         (doto mln .updateRnnStateWithTBPTTState)))

(defn z-from-prev-layer
  "Compute input linear transformation (z) from previous layer
  Applies pre processing transformation where necessary

  :current-layer-idx (int), the current layer

  :input (INDArray or vec), the input

  :training? (boolean), are we in training mode?

  returns the activation from the previous layer"
  [& {:keys [mln current-layer-idx input training? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :current-layer-idx (:or (_ :guard number?)
                                   (_ :guard seq?))
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :training (:or (_ :guard boolean?)
                          (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.zFromPrevLayer ~mln (int ~current-layer-idx)
                           (vec-or-matrix->indarray ~input) ~training?))
         :else
         (let [[model cur-idx i t?] (eval-if-code [mln seq?] [current-layer-idx seq? number?]
                                                  [input seq?] [training? seq? boolean?])]
           (.zFromPrevLayer model cur-idx (vec-or-matrix->indarray i) t?))))
