# Find Graphviz
# Defines:
#  GRAPHVIZ_FOUND
#  GRAPHVIZ_INCLUDE_DIR
#  GRAPHVIZ_LIBRARY
#  GRAPHVIZ_DEFINITIONS

IF (GRAPHVIZ_INCLUDE_DIR AND GRAPHVIZ_CDT_LIBRARY AND GRAPHVIZ_CGRAPH_LIBRARY AND GRAPHVIZ_GRAPH_LIBRARY AND GRAPHVIZ_PATHPLAN_LIBRARY)
    SET(GRAPHVIZ_FIND_QUIETLY TRUE)
ENDIF (GRAPHVIZ_INCLUDE_DIR AND GRAPHVIZ_CDT_LIBRARY AND GRAPHVIZ_CGRAPH_LIBRARY AND GRAPHVIZ_GRAPH_LIBRARY AND GRAPHVIZ_PATHPLAN_LIBRARY)

FIND_PATH( GRAPHVIZ_INCLUDE_DIR graphviz/graph.h)

FIND_LIBRARY( GRAPHVIZ_CDT_LIBRARY NAMES cdt )
FIND_LIBRARY( GRAPHVIZ_GVC_LIBRARY NAMES gvc )
FIND_LIBRARY( GRAPHVIZ_CGRAPH_LIBRARY NAMES cgraph )
FIND_LIBRARY( GRAPHVIZ_GRAPH_LIBRARY NAMES graph )
FIND_LIBRARY( GRAPHVIZ_PATHPLAN_LIBRARY NAMES pathplan )

IF (GRAPHVIZ_INCLUDE_DIR AND GRAPHVIZ_CDT_LIBRARY AND GRAPHVIZ_GVC_LIBRARY AND GRAPHVIZ_CGRAPH_LIBRARY AND GRAPHVIZ_GRAPH_LIBRARY AND GRAPHVIZ_PATHPLAN_LIBRARY)
   SET(GRAPHVIZ_FOUND TRUE)
ELSE (GRAPHVIZ_INCLUDE_DIR AND GRAPHVIZ_CDT_LIBRARY AND GRAPHVIZ_GVC_LIBRARY AND GRAPHVIZ_CGRAPH_LIBRARY AND GRAPHVIZ_GRAPH_LIBRARY AND GRAPHVIZ_PATHPLAN_LIBRARY)
   SET(GRAPHVIZ_FOUND FALSE)
ENDIF (GRAPHVIZ_INCLUDE_DIR AND GRAPHVIZ_CDT_LIBRARY AND GRAPHVIZ_GVC_LIBRARY AND GRAPHVIZ_CGRAPH_LIBRARY AND GRAPHVIZ_GRAPH_LIBRARY AND GRAPHVIZ_PATHPLAN_LIBRARY)

IF (GRAPHVIZ_FOUND)
  SET(GRAPHVIZ_LIBRARY "${GRAPHVIZ_CDT_LIBRARY} ${GRAPHVIZ_GVC_LIBRARY} ${GRAPHVIZ_CGRAPH_LIBRARY} ${GRAPHVIZ_GRAPH_LIBRARY} ${GRAPHVIZ_PATHPLAN_LIBRARY}")
  IF (NOT GRAPHVIZ_FIND_QUIETLY)
    MESSAGE(STATUS "Found Graphviz: ${GRAPHVIZ_LIBRARY}")
  ENDIF (NOT GRAPHVIZ_FIND_QUIETLY)
ELSE (GRAPHVIZ_FOUND)
  IF (GRAPHVIZ_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could NOT find Graphivz")
  ENDIF (GRAPHVIZ_FIND_REQUIRED)
ENDIF (GRAPHVIZ_FOUND)

