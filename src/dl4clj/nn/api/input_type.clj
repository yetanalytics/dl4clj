(ns dl4clj.nn.api.input-type
  (:import [org.deeplearning4j.nn.conf.layers InputTypeUtil])
  (:require [dl4clj.constants :as enum]
            [dl4clj.utils :refer [obj-or-code?]]
            [clojure.core.match :refer [match]]))

(defn get-output-type-cnn-layers
  "returns the output type from cnn layers

  :input-type (map), the input to the cnn layer
   - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants

  :kernel-size (vec) dimensions of the kernel

  :stride (vec), the stride for the cnn layer

  :padding (vec), the padding for the cnn layer

  :convolution-mode (keyword), defines how convolution operations should be executed
   - one of: :strict, :truncate, :same
   - see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ConvolutionMode.html

  :output-depth (int), the output depth

  :layer-idx (int), the index of the layer within the model

  :layer-name (str), the name given for the layer

  :layer-class (java class), the java class for the layer"
  [& {:keys [input-type kernel-size
             stride padding convolution-mode
             output-depth layer-idx layer-name
             layer-class as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:input-type (:or (_ :guard map?)
                            (_ :guard seq?))
           :kernel-size (:or (_ :guard vector?)
                             (_ :guard seq?))
           :stride (:or (_ :guard vector?)
                        (_ :guard seq?))
           :padding (:or (_ :guard vector?)
                         (_ :guard seq?))
           :convolution-mode (:or (_ :guard keyword?)
                                  (_ :guard seq?))
           :output-depth (:or (_ :guard number?)
                              (_ :guard seq?))
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))
           :layer-name (:or (_ :guard string?)
                            (_ :guard seq?))
           :layer-class _}]
         (obj-or-code?
          as-code?
          `(InputTypeUtil/getOutputTypeCnnLayers
           (enum/input-types ~input-type)
           (int-array ~kernel-size)
           (int-array ~stride)
           (int-array ~padding)
           (enum/value-of {:convolution-mode ~convolution-mode})
           ~output-depth ~layer-idx ~layer-name ~layer-class))
         :else
         (InputTypeUtil/getOutputTypeCnnLayers
          (enum/input-types input-type)
          (int-array kernel-size)
          (int-array stride)
          (int-array padding)
          (enum/value-of {:convolution-mode convolution-mode})
          output-depth layer-idx layer-name layer-class)))

(defn get-pre-processor-for-input-type-cnn-layers
  "Utility method for determining the appropriate preprocessor for CNN layers

  :input-type (map), the input to the cnn layer
  - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants

  :layer-name (str), the name of the layer

  if no preprocessor is required, will return nil"
  [& {:keys [input-type layer-name as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:input-type (:or (_ :guard map?)
                            (_ :guard seq?))
           :layer-name (:or (_ :guard string?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(InputTypeUtil/getPreProcessorForInputTypeCnnLayers
           (enum/input-types ~input-type) ~layer-name))
         :else
         (InputTypeUtil/getPreProcessorForInputTypeCnnLayers
          (enum/input-types input-type) layer-name)))

(defn get-pre-processor-for-input-type-rnn-layers
  "Utility method for determining the appropriate preprocessor for recurrent layers

  :input-type (map), the input to the cnn layer
  - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants

  :layer-name (str), the name of the layer

  if no preprocessor is required, will return nil"
  [& {:keys [input-type layer-name as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:input-type (:or (_ :guard map?)
                            (_ :guard seq?))
           :layer-name (:or (_ :guard string?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(InputTypeUtil/getPreprocessorForInputTypeRnnLayers
           (enum/input-types ~input-type) ~layer-name))
         :else
         (InputTypeUtil/getPreprocessorForInputTypeRnnLayers
          (enum/input-types input-type) layer-name)))
