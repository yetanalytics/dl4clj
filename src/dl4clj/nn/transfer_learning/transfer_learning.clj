(ns ^{:doc "The transfer learning API can be used to modify the architecture or the learning parameters of an existing multilayernetwork or computation graph.
 It allows one to - change nOut of an existing layer - remove and add existing layers/vertices - fine tune learning configuration (learning rate, updater etc) - hold parameters for specified layers as a constant

see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/transferlearning/TransferLearning.htmlq"}
    dl4clj.nn.transfer-learning.transfer-learning
  (:import [org.deeplearning4j.nn.transferlearning
            TransferLearning
            ;; currently not supporting graph-builder
            TransferLearning$Builder])
  (:require [dl4clj.nn.conf.builders.layers :as layer]
            [dl4clj.nn.conf.input-pre-processor :as pre-process]
            [clojure.core.match :refer [match]]
            [dl4clj.nn.api.model :refer [init!]]
            [dl4clj.nn.api.multi-layer-network :refer [is-init-called?]]
            [dl4clj.utils :refer [builder-fn eval-and-build replace-map-vals generic-dispatching-fn]]
            [dl4clj.helpers :refer [value-of-helper distribution-helper pre-processor-helper]]))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; helper fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn replace-layer-helper
  "performs the replacement of n-out for a supplied layer

  This fn is called within multi-layer-network-mutater-builder

  :layer-idx (int), index of the layer you want to change

  :n-out (int), desired n-out for the supplied layer

  :dist (distribution), a distribution to sample weights from
   - see: dl4clj.nn.conf.distribution.distribution

  :next-dist (distribution), distribtuion to use for params in the next layer
   - see: dl4clj.nn.conf.distribution.distribution

  :weight-init (keyword) Weight initialization scheme
  one of: :distribution, :zero, :sigmoid-uniform, :uniform, :xavier, :xavier-uniform
          :xavier-fan-in, :xavier-legacy, :relu, :relu-uniform, :vi, :size

  :next-weight-init (keyword), same as :weight-init but applied to the next layer"
  [{:keys [layer-idx n-out dist next-dist
           weight-init next-weight-init]
    :as opts}]
  (let [d (if dist
            (distribution-helper dist))
        next-d (if next-dist
                 (distribution-helper next-dist))
        w (if weight-init
            (value-of-helper :weight-init weight-init))
        next-w (if next-weight-init
                 (value-of-helper :weight-init next-weight-init))]
    (match [opts]
           [{:layer-idx _ :n-out _ :dist _ :weight-init _}]
           `[~layer-idx ~n-out ~d ~w]
           [{:layer-idx _ :n-out _ :dist _ :next-weight-init _}]
           `[~layer-idx ~n-out ~d ~next-w]
           [{:layer-idx _ :n-out _ :weight-init _ :next-dist _}]
           `[~layer-idx ~n-out ~w ~next-d]
           [{:layer-idx _ :n-out _ :dist _ :next-dist _}]
           `[~layer-idx ~n-out ~d ~next-d]
           [{:layer-idx _ :n-out _ :weight-init _ :next-dist _}]
           `[~layer-idx ~n-out ~w ~next-d]
           [{:layer-idx _ :n-out _ :weight-init _ :next-weight-init _}]
           `[~layer-idx ~n-out ~w ~next-w]
           [{:layer-idx _ :n-out _ :dist _}]
           `[~layer-idx ~n-out ~d]
           [{:layer-idx _ :n-out _ :weight-init _}]
           `[~layer-idx ~n-out ~w]
           :else
           (assert false "missing args, you need atleast a distribution or a weight-init"))))

(defn input-pre-processor-helper
  "addds a preprocessor for a given layer

  this fn is called by multi-layer-network-mutater-builder

  :layer-idx (int), index of the layer you want to mutate

  :pre-processor (pre-processor), a built pre-processor
   - see: dl4clj.nn.conf.input-pre-processor"
  [{:keys [layer-idx pre-processor]
    :as opts}]
  `[~layer-idx
    ~(match [pre-processor]
            [(_ :guard seq?)] pre-processor
            :else
            `(pre-process/pre-processors ~pre-processor))])

(defn add-layers-helper
  "adds multiple layers to a transfer learning builder

  maintains order of int keys in the passed in layers map when

  multiple layers are suppose to be added"
  [layers]
  (match [layers]
         ;; sinle layer, fn call
         [(_ :guard seq?)]
         [`(eval-and-build ~layers)]
         ;; single layer, config map
         [(true :<< #(keyword? (generic-dispatching-fn %)))]
         [`(eval-and-build (layer/builder ~layers))]
         ;; multiple layers
         [(true :<< #(integer? (generic-dispatching-fn %)))]
         ;; ensure ordering based on passed in idxs
         (let [idxs (sort (keys layers))
               ls (into []
                        (for [each idxs]
                          (get layers each)))]
           (loop [result []
                  cur! ls]
             ;; tail recursion
             (cond (empty? cur!)
                   result
                   :else
                   (let [cur-layer (first cur!)]
                     (recur (conj result
                                  (match [cur-layer]
                                         ;; are we dealing with a config map or a fn call
                                         [(_ :guard map?)]
                                         [`(eval-and-build
                                            (layer/builder ~cur-layer))]
                                         :else
                                         [`(eval-and-build ~cur-layer)]))
                            (rest cur!))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; builder for mutating multi layer networks
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def tlb-method-map
  {:fine-tune-conf            '.fineTuneConfiguration
   :remove-last-n-layers      '.removeLayersFromOutput
   :remove-output-layer?      '.removeOutputLayer
   :add-layers                '.addLayer
   :set-feature-extractor-idx '.setFeatureExtractor
   :replacement-layer         '.nOutReplace
   :input-pre-processor       '.setInputPreProcessor})

(defn builder
  "given a multi-layer-network and options on how to change it,

  creates a builder for applying the changes and applies them.

  returns the mutated multi layer network (mln)

  :mln (multi layer network), a multi layer network to mutate
   - see dl4clj.nn.conf.builders.multi-layer-builders

  :add-layers (multiple layers), a map of layers to add to the net, will be added
   in the order of their supplied index

  :fine-tune-conf (fine tune configuration), a fine tune configuration
   - required to add layer(s)
   - see dl4clj.nn.transfer-learning.fine-tune-conf
   - result of new-fine-tune-conf when eval-and-build? set to false

  :replacement-layer (map), {:layer-idx (int) :n-out (int) :dist (distribution)
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
   - see: dl4clj.nn.conf.input-pre-processor"
  [& {:keys [mln fine-tune-conf input-pre-processor remove-last-n-layers
             replacement-layer set-feature-extractor-idx add-layers
             remove-output-layer? as-code? tlb]
      :or {as-code? true}
      :as opts}]
  (let [b (if tlb
            tlb
           `(TransferLearning$Builder.  (if (is-init-called? ~mln)
                                        ~mln
                                        (init! :model ~mln))))

        replacement-layer* (if replacement-layer
                             (replace-layer-helper replacement-layer))

        input-pre-processor* (if input-pre-processor
                               ;; should I account for multiple input pre processors here?
                               (input-pre-processor-helper input-pre-processor))

        add-layers* (if add-layers
                      (add-layers-helper add-layers))

        fine-tune-conf* (if fine-tune-conf
                          `(eval-and-build ~fine-tune-conf))

        updated-opts {:replacement-layer replacement-layer*
                      :input-pre-processor input-pre-processor*
                      :fine-tune-conf fine-tune-conf*}

        opts* (replace-map-vals (dissoc opts :mln :as-code? :remove-output-layer?
                                        :add-layers :tlb)
                                updated-opts)

        fn-chain (builder-fn b tlb-method-map opts*)

        ;; layers have to be added after the fine tune conf

        fn-chain-output-removed (if remove-output-layer?
                                  `(.removeOutputLayer ~fn-chain)
                                  fn-chain)


        fn-chain* (if add-layers*
                    (builder-fn fn-chain-output-removed tlb-method-map {:add-layers add-layers*})
                    fn-chain-output-removed)]
    (if as-code?
      fn-chain*
      (eval-and-build fn-chain*))))
