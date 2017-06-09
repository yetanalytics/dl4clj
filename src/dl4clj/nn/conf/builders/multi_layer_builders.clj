(ns dl4clj.nn.conf.builders.multi-layer-builders
  (:require [dl4clj.nn.conf.builders.builders :as bb]
            [dl4clj.nn.conf.input-pre-processor :as pre-process]
            [dl4clj.nn.conf.constants :as constants])
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder
            NeuralNetConfiguration$ListBuilder MultiLayerConfiguration$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]))

(defn multi-layer-config-builder
  "creates and builds a multi layer neural network configuration.

  you must either supply :list-builder or :confs
   - you need some configuration to build from
   - if only :confs is supplied, a fresh multi-layer-conf-builder will be used
   - if list builder is supplied, that builder will have these params applied to it

  params are:

  :list-builder (nn-conf-list-builder) the result of nn-conf-builder
   - when you supply the :layers key and :built? is set to false (default)

  :confs (coll), a collection of built neural net configurations
   - needs to be single layer neural network configurations
   - when you call nn-conf-builder and supply :layer and :build? true

  :backprop? (boolean) whether to do backprop or not

  :backprop-type (keyword) the type of backprop, one of :standard or :truncated-bptt

  :input-pre-processors {int pre-processor} ie {0 (new-zero-mean-pre-pre-processor)
                                                1 (new-unit-variance-processor)}
  specifies the processors, these are used at each layer for doing things like
  normalization and shaping of input.
   - the pre-processor can be the obj created by the new- fns in input-pre-processor
     or a config map for creating the desired pre-processor

  :input-type (map), map of params describing the input
   {(keyword) other-opts}, ie. {:convolutional {:input-height 1 ...}}
    - the first key is one of: :convolutional, :convolutional-flat, :feed-forward, :recurrent
    - other opts: for feedforward and recurrent, supply the :size of the layer
                  for convolutional, supply the height width depth of the layer

  :pretrain? (boolean) Whether to do pre train or not

  :tbptt-back-length (int) When doing truncated BPTT: how many steps of backward should we do?
  Only applicable when :backprop-type = :truncated-bptt

  :tbptt-fwd-length (int) When doing truncated BPTT: how many steps of forward pass
  should we do before doing (truncated) backprop? Only applicable when :backprop-type = :truncated-bptt
   - Typically tBPTTForwardLength parameter is same as the tBPTTBackwardLength parameter,
     but may be larger than it in some circumstances (but never smaller)
   - Ideally your training data time series length should be divisible by this"
  [& {:keys [list-builder nn-confs backprop? backprop-type damping-factor
             input-pre-processors input-type pretrain?
             tbptt-back-length tbptt-fwd-length build?]
      :or {build? true}
      :as opts}]
  (let [b (if (contains? opts :list-builder)
            list-builder
            (MultiLayerConfiguration$Builder.))
        confz (if (coll? nn-confs)
                (into '() nn-confs)
                (into '() [nn-confs]))
        pps (into {}
                  (for [each input-pre-processors
                        :let [[idx pp] each]]
                    (if (map? pp)
                      {idx (pre-process/pre-processors pp)}
                      {idx pp})))]
    (cond-> b
      (contains? opts :backprop?) (.backprop backprop?)
      (contains? opts :backprop-type) (.backpropType (constants/value-of
                                                      {:backprop-type
                                                       backprop-type}))
      (contains? opts :nn-confs) (.confs confz)
      (contains? opts :damping-factor) (.dampingFactor damping-factor)
      (contains? opts :input-pre-processors) (.inputPreProcessors pps)
      (contains? opts :input-type) (.setInputType (constants/input-types input-type))
      (contains? opts :pretrain?) (.pretrain pretrain?)
      (contains? opts :tbptt-back-length) (.tBPTTBackwardLength tbptt-back-length)
      (contains? opts :tbptt-fwd-length) (.tBPTTForwardLength tbptt-fwd-length)
      (true? build?) (.build))))

(defn list-builder
  ;; could refactor to be recursive (no counters)
  "builds a list of layers to be used in a multi-layer configuration

  layers should take the form of {:layers {idx {:layer-type layer-opts}}}
  or {:layers {idx (layer-type-builder opts)}}
  (layer-type-builders are found at the bottom of the builders ns)

  :layers can also be a mix of param maps and function calls.  ie,
  {:layers {0 {:activation-layer {opts-keys opts-values}}
            1 (activation-layer-builder {opts-keys opts-values})}}

  see the builders ns for layer opts"
  [nn-conf-builder layers]
  (let [b (.list nn-conf-builder)
        max-idx (+ 1 (last (sort (map first layers))))]
    (loop [idx 0
           result b]
      (cond (not= idx max-idx)
            (let [current-layer (get layers idx)]
              ;; guarantees layers are built in the order specified by the idxs
              ;; maps are not inherently ordered
              (if (map? current-layer)
                ;; we are dealing with a config map that needs to go through builder multimethod
                (recur
                 (inc idx)
                 (.layer result idx (bb/builder current-layer)))
                ;; we have already been through the builder multimethod and just need to set the layer
                (recur
                 (inc idx)
                 (.layer result idx current-layer))))
            (= idx max-idx)
            result))))
