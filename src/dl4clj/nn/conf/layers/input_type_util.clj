(ns dl4clj.nn.conf.layers.input-type-util
  (:import [org.deeplearning4j.nn.conf.layers InputTypeUtil]))

(defn get-output-type-cnn-layers
  [& {:keys [input-type kernel-size
             stride padding convolution-mode
             output-depth layer-idx layer-name
             layer-class]}]
  (.getOutputTypeCnnLayers input-type kernel-size
                           stride padding convolution-mode
                           output-depth layer-idx layer-name
                           layer-class))

(defn get-pre-processor-for-input-type-cnn-layers
  [& {:keys [input-type layer-name]}]
  (.getPreProcessorForInputTypeCnnLayers input-type layer-name))

(defn get-pre-processor-for-input-type-rnn-layers
  [& {:keys [input-type layer-name]}]
  (.getPreprocessorForInputTypeRnnLayers input-type layer-name))
