(ns dl4clj.streaming.routes.dl4j-serve-route-builder
  (:import [org.deeplearning4j.streaming.routes DL4jServeRouteBuilder]))

;; it looks like i need to call the camel methods for setting up the builder?
;; no idea, cant call constructor...
;; maybe I dont have the camel dep?
;; will need to look at the test classes they set up

#_(defn new-dl4j-serve-route-builder
  "Serve results from a kafka queue.
  The input to the route can either be a pre serialized ndarray
  or a normal ndarray itself."
  []
  (DL4jServeRouteBuilder.))
