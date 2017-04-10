(ns dl4clj.nn.api.param-initializer
  (:import [org.deeplearning4j.nn.api ParamInitializer]))

(defn get-gradients-from-flattened
  "Return a map of gradients (in their standard non-flattened representation),
  taken from the flattened (row vector) gradientView array."
  [& {:keys [this conf gradient-view]}]
  (.getGradientsFromFlattened this conf gradient-view))

(defn init
  "Initialize the parameters"
  [& {:keys [this conf params-view initialize-params?]}]
  (.init this conf params-view initialize-params?))

(defn num-params
  "return number of params"
  [& {:keys [this conf]}]
  (.numParams this conf))
