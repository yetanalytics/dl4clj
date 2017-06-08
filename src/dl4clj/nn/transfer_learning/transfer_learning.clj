(ns ^{:doc "The transfer learning API can be used to modify the architecture or the learning parameters of an existing multilayernetwork or computation graph.
 It allows one to - change nOut of an existing layer - remove and add existing layers/vertices - fine tune learning configuration (learning rate, updater etc) - hold parameters for specified layers as a constant

see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/transferlearning/TransferLearning.html"}
    dl4clj.nn.transfer-learning.transfer-learning
  (:import [org.deeplearning4j.nn.transferlearning
            TransferLearning
            ;; currently not supporting graph-builder
            TransferLearning$Builder])
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.constants :as enum]
            [dl4clj.nn.conf.builders.builders :as l]))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; helper fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn replace-n-out
  "performs the replacement of n-out for a supplied layer

  This fn is called within multi-layer-network-mutater-builder

  :layer-idx (int), index of the layer you want to change

  :n-out (int), desired n-out for the supplied layer

  :dist (distribution), a distribution to sample weights from
   - creation of the distribution is not handled here
   - see: dl4clj.nn.conf.distribution.distribution

  :next-dist (distribution), distribtuion to use for params in the next layer
   - creation of the distribution is not handled here
   - see: dl4clj.nn.conf.distribution.distribution

  :weight-init (keyword) Weight initialization scheme
  one of: :distribution, :zero, :sigmoid-uniform, :uniform, :xavier, :xavier-uniform
          :xavier-fan-in, :xavier-legacy, :relu, :relu-uniform, :vi, :size

  :next-weight-init (keyword), same as :weight-init but applied to the next layer"
  [builder {:keys [layer-idx n-out dist next-dist
                   weight-init next-weight-init]
            :as opts}]
  (assert (contains-many? opts :layer-idx :n-out)
          "You must supply a layer index and the n-out for replacement")
  (cond (contains-many? opts :dist :next-weight-init)
        (.nOutReplace builder layer-idx n-out dist (enum/value-of
                                                    {:weight-init next-weight-init}))
        (contains-many? opts :dist :next-dist)
        (.nOutReplace builder layer-idx n-out dist next-dist)
        (contains-many? opts :weight-init :next-dist)
        (.nOutReplace builder layer-idx n-out (enum/value-of
                                               {:weight-init weight-init}) next-dist)
        (contains-many? opts :weight-init :next-weight-init)
        (.nOutReplace builder layer-idx n-out
                      (enum/value-of
                       {:weight-init weight-init})
                      (enum/value-of
                       {:weight-init next-weight-init}))
        (contains-many? opts :dist)
        (.nOutReplace builder layer-idx n-out dist)
        (contains-many? opts :weight-init)
        (.nOutReplace builder layer-idx n-out (enum/value-of
                                               {:weight-init weight-init}))
        :else
        (assert false "no mutation happened, you need atleast a distribution or a weight-init")))

(defn set-input-pre-processor!
  "addds a preprocessor for a given layer

  this fn is called by multi-layer-network-mutater-builder

  :layer-idx (int), index of the layer you want to mutate

  :pre-processor (pre-processor), a built pre-processor
   - see: dl4clj.nn.conf.input-pre-processor"
  [builder {:keys [layer-idx pre-processor]
            :as opts}]
  (.setInputPreProcessor builder layer-idx pre-processor))

(defn add-multiple-layers
  "adds multiple layers to a transfer learning builder"
  [tl-builder layers]
  (let [idxs (sort (keys layers))
        layers (into []
                     (for [each idxs]
                       (get layers each)))]
    (loop [result tl-builder
           cur! layers]
      (cond (empty? cur!)
            result
            :else
            (let [cur-layer (first cur!)]
              (recur (.addLayer result (if (map? cur-layer)
                                         (l/builder cur-layer)
                                         cur-layer))
                     (rest cur!)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; builder for mutating multi layer networks
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn transfer-learning-builder
  "given a multi-layer-network and options on how to change it,

  creates a builder for applying the changes and applies them.

  returns the mutated multi layer network (mln)

  :mln (multi layer network), a multi layer network to mutate
   - see dl4clj.nn.conf.builders.multi-layer-builders

  :tlb (transferlearning builder), an existing transfer learning builder
   - if not supplied, a new one will be created from the mln

  :add-layer (layer), a layer to add to the multi layer network
   - see dl4clj.nn.conf.builders.builders

  :add-layers (multiple layers), a map of layers to add to the net, will be added
   in the order of their supplied index
    - you should either add a single layer by supplying :layer and not :layers or
      multiple layers by supplying this key

  :fine-tune-conf (fine tune configuration), a fine tune configuration
   - required to add layer(s)
   - see dl4clj.nn.transfer-learning.fine-tune-conf

  :n-out-replace (map), {:layer-idx (int) :n-out (int) :dist (distribution)
                         :next-dist (distribution) :weight-init (keyword)
                         :next-weight-init (keyword)}
   - see replace-n-out for further details about the args
   - see dl4clj.nn.conf.distribution.distribution for creating distributions

  :remove-last-n-layers (int), Remove last n layers of the net
   - At least an output layer must be added back in

  :remove-output-layer? (boolean), if true, will remove the output layer from the MLN

  :set-feature-extractor-idx (int), Specify a layer to set as a feature extractor
   - The specified layer and the layers preceding it will be frozen with parameters staying constant

  :input-pre-processor (map) {:layer-idx (int) :pre-processor (pre-processor)}
   - see: dl4clj.nn.conf.input-pre-processor

  :build? (boolean), wether or not to build the mutation, defaults to true"
  [& {:keys [mln add-layer add-layers fine-tune-conf
           n-out-replace remove-last-n-layers
           remove-output-layer? set-feature-extractor-idx
           input-pre-processor build? tlb]
      :or {build? true}
    :as opts}]
  (let [b (if (contains? opts :tlb)
            tlb
           (TransferLearning$Builder. mln))]
    (cond-> b
      (contains? opts :fine-tune-conf) (.fineTuneConfiguration fine-tune-conf)
      (contains? opts :n-out-replace) (replace-n-out n-out-replace)
      (contains? opts :remove-last-n-layers) (.removeLayersFromOutput remove-last-n-layers)
      (true? remove-output-layer?) .removeOutputLayer
      ;; refactor so can add multiple layers
      (contains? opts :add-layer) (.addLayer add-layer)
      (contains? opts :add-layers) (add-multiple-layers add-layers)
      (contains? opts :set-feature-extractor-idx) (.setFeatureExtractor set-feature-extractor-idx)
      (contains? opts :input-pre-processor) (set-input-pre-processor! input-pre-processor)
      (true? build?) .build)))
