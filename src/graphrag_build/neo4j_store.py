"""
neo4j_store.py — Neo4j Graph Database backend cho GraphRAG KG.

Cho phép import NetworkX DiGraph (entities + relationships) vào Neo4j,
và cung cấp các query helper để RAG pipeline truy vấn subgraph.

Usage:
    store = Neo4jStore("bolt://localhost:7687", "neo4j", "graphrag2026")
    store.clear_graph()
    store.import_from_networkx(kg, all_entities, all_relationships)
    store.close()
"""

import logging
from typing import Any, Dict, List, Optional, Set

import networkx as nx

logger = logging.getLogger("NEO4J_STORE")

# Batch size cho UNWIND import — tránh transaction quá lớn
_BATCH_SIZE = 500


class Neo4jStore:
    """Quản lý kết nối và import/export KG với Neo4j."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "graphrag2026",
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j driver chưa được cài. Chạy: pip install neo4j>=5.0"
            )

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        logger.info(f"✅ Neo4j connected: {uri}")

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Đóng connection."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ── Schema & Cleanup ─────────────────────────────────────────────────

    def create_indexes(self) -> None:
        """Tạo index trên Entity.name để tăng tốc lookup."""
        with self._driver.session() as session:
            session.run(
                "CREATE INDEX entity_name_idx IF NOT EXISTS "
                "FOR (e:Entity) ON (e.name)"
            )
            session.run(
                "CREATE INDEX entity_type_idx IF NOT EXISTS "
                "FOR (e:Entity) ON (e.type)"
            )
        logger.info("✅ Neo4j indexes created.")

    def clear_graph(self) -> None:
        """Xóa toàn bộ nodes & edges trong database (dùng khi rebuild)."""
        with self._driver.session() as session:
            # Xóa theo batch để tránh OOM với graph lớn
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS deleted"
                )
                deleted = result.single()["deleted"]
                if deleted == 0:
                    break
        logger.info("🗑️  Neo4j graph cleared.")

    # ── Import ───────────────────────────────────────────────────────────

    def import_from_networkx(
        self,
        kg: nx.DiGraph,
        all_entities: Optional[List[Dict[str, Any]]] = None,
        all_relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Import KG từ NetworkX DiGraph vào Neo4j.

        Ưu tiên dùng dữ liệu từ NetworkX graph (nodes + edges).
        Nếu all_entities / all_relationships được cung cấp thêm,
        sẽ merge thông tin bổ sung.
        """
        self.create_indexes()

        # ── Import nodes ──
        nodes_data = []
        for node, attrs in kg.nodes(data=True):
            nodes_data.append({
                "name": str(node),
                "type": attrs.get("type", "UNKNOWN"),
                "description": attrs.get("description", ""),
                "source_chunks": list(attrs.get("source_chunks", [])),
                "frequency": attrs.get("frequency", 1),
            })

        self._batch_import_nodes(nodes_data)

        # ── Import edges ──
        edges_data = []
        for src, tgt, attrs in kg.edges(data=True):
            edges_data.append({
                "source": str(src),
                "target": str(tgt),
                "relation": attrs.get("relation", "liên_quan"),
                "description": attrs.get("description", ""),
                "weight": attrs.get("weight", 0.5),
            })

        self._batch_import_edges(edges_data)

        logger.info(
            f"✅ Neo4j import done: {len(nodes_data)} nodes, {len(edges_data)} edges"
        )

    def _batch_import_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        """Import nodes theo batch dùng UNWIND."""
        with self._driver.session() as session:
            for i in range(0, len(nodes), _BATCH_SIZE):
                batch = nodes[i : i + _BATCH_SIZE]
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (e:Entity {name: row.name})
                    SET e.type = row.type,
                        e.description = row.description,
                        e.source_chunks = row.source_chunks,
                        e.frequency = row.frequency
                    """,
                    batch=batch,
                )
                logger.info(f"   nodes batch {i // _BATCH_SIZE + 1}: {len(batch)} imported")

    def _batch_import_edges(self, edges: List[Dict[str, Any]]) -> None:
        """Import edges theo batch dùng UNWIND."""
        with self._driver.session() as session:
            for i in range(0, len(edges), _BATCH_SIZE):
                batch = edges[i : i + _BATCH_SIZE]
                session.run(
                    """
                    UNWIND $batch AS row
                    MATCH (s:Entity {name: row.source})
                    MATCH (t:Entity {name: row.target})
                    MERGE (s)-[r:RELATES_TO {relation: row.relation}]->(t)
                    SET r.description = row.description,
                        r.weight = row.weight
                    """,
                    batch=batch,
                )
                logger.info(f"   edges batch {i // _BATCH_SIZE + 1}: {len(batch)} imported")

    # ── Query helpers (cho RAG) ──────────────────────────────────────────

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin 1 entity theo tên."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {name: $name}) RETURN e", name=name
            )
            record = result.single()
            if record:
                node = record["e"]
                return dict(node)
        return None

    def query_neighbors(
        self, entity_name: str, depth: int = 1, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Lấy neighbors của 1 entity trong phạm vi depth hops.

        Returns list of dicts: {source, target, relation, weight, description}
        """
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (start:Entity {{name: $name}})
                MATCH path = (start)-[r:RELATES_TO*1..{depth}]-(neighbor:Entity)
                UNWIND relationships(path) AS rel
                WITH startNode(rel) AS s, endNode(rel) AS t, rel
                RETURN DISTINCT
                    s.name AS source,
                    t.name AS target,
                    rel.relation AS relation,
                    rel.weight AS weight,
                    rel.description AS description
                LIMIT $limit
                """,
                name=entity_name,
                limit=limit,
            )
            return [dict(record) for record in result]

    def get_subgraph(
        self, entity_names: List[str], depth: int = 1
    ) -> Dict[str, Any]:
        """
        Lấy subgraph chứa các entities đã cho và neighbors.

        Returns: {"nodes": [...], "edges": [...]}
        """
        with self._driver.session() as session:
            result = session.run(
                f"""
                UNWIND $names AS ename
                MATCH (start:Entity {{name: ename}})
                OPTIONAL MATCH path = (start)-[r:RELATES_TO*1..{depth}]-(neighbor:Entity)
                WITH collect(DISTINCT start) + collect(DISTINCT neighbor) AS all_nodes
                UNWIND all_nodes AS n
                WITH collect(DISTINCT n) AS nodes
                UNWIND nodes AS n1
                UNWIND nodes AS n2
                OPTIONAL MATCH (n1)-[r:RELATES_TO]->(n2)
                WHERE r IS NOT NULL
                WITH nodes,
                     collect(DISTINCT {{
                         source: n1.name,
                         target: n2.name,
                         relation: r.relation,
                         weight: r.weight,
                         description: r.description
                     }}) AS edges
                RETURN
                    [n IN nodes | {{
                        name: n.name,
                        type: n.type,
                        description: n.description
                    }}] AS nodes,
                    edges
                """,
                names=entity_names,
            )
            record = result.single()
            if record:
                return {
                    "nodes": record["nodes"],
                    "edges": [e for e in record["edges"] if e["source"] is not None],
                }
        return {"nodes": [], "edges": []}

    def get_stats(self) -> Dict[str, int]:
        """Lấy thống kê cơ bản: số nodes, edges."""
        with self._driver.session() as session:
            nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            edges = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
        return {"nodes": nodes, "edges": edges}
