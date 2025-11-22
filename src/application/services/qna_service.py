"""
Q&A service for question and answer management
PostgreSQL 버전
"""

from typing import List, Dict, Optional
from fastapi import HTTPException
from src.infrastructure.persistence.database import get_db_connection, release_connection, get_cursor


def get_questions_for_festival(
    festival_name: str,
    limit: int = 50,
    offset: int = 0
) -> List[Dict]:
    """Get all questions for a specific festival"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute(
            """
            SELECT
                q.id, q.festival_name, q.user_id, q.title, q.content, q.views,
                q.created_at, q.updated_at,
                u.username, u.full_name,
                (SELECT COUNT(*) FROM answers WHERE question_id = q.id) as answer_count
            FROM questions q
            JOIN users u ON q.user_id = u.id
            WHERE q.festival_name = %s
            ORDER BY q.created_at DESC
            LIMIT %s OFFSET %s
            """,
            (festival_name, limit, offset),
        )
        rows = cursor.fetchall()

        questions = []
        for row in rows:
            questions.append({
                "id": row["id"],
                "festival_name": row["festival_name"],
                "user_id": row["user_id"],
                "title": row["title"],
                "content": row["content"],
                "views": row["views"],
                "created_at": str(row["created_at"]) if row["created_at"] else None,
                "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
                "author": {
                    "username": row["username"],
                    "full_name": row["full_name"],
                },
                "answer_count": row["answer_count"],
            })
        return questions
    finally:
        cursor.close()
        release_connection(conn)


def get_question_by_id(question_id: int) -> Optional[Dict]:
    """Get a single question with its answers"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        # Increment view count
        cursor.execute(
            "UPDATE questions SET views = views + 1 WHERE id = %s",
            (question_id,),
        )
        conn.commit()

        # Get question
        cursor.execute(
            """
            SELECT
                q.id, q.festival_name, q.user_id, q.title, q.content, q.views,
                q.created_at, q.updated_at,
                u.username, u.full_name
            FROM questions q
            JOIN users u ON q.user_id = u.id
            WHERE q.id = %s
            """,
            (question_id,),
        )
        q_row = cursor.fetchone()

        if not q_row:
            return None

        # Get answers
        cursor.execute(
            """
            SELECT
                a.id, a.question_id, a.user_id, a.content, a.is_accepted,
                a.created_at, a.updated_at,
                u.username, u.full_name, u.role
            FROM answers a
            JOIN users u ON a.user_id = u.id
            WHERE a.question_id = %s
            ORDER BY a.is_accepted DESC, a.created_at ASC
            """,
            (question_id,),
        )
        answer_rows = cursor.fetchall()

        answers = []
        for a_row in answer_rows:
            answers.append({
                "id": a_row["id"],
                "question_id": a_row["question_id"],
                "user_id": a_row["user_id"],
                "content": a_row["content"],
                "is_accepted": bool(a_row["is_accepted"]),
                "created_at": str(a_row["created_at"]) if a_row["created_at"] else None,
                "updated_at": str(a_row["updated_at"]) if a_row["updated_at"] else None,
                "author": {
                    "username": a_row["username"],
                    "full_name": a_row["full_name"],
                    "role": a_row["role"],
                },
            })

        return {
            "id": q_row["id"],
            "festival_name": q_row["festival_name"],
            "user_id": q_row["user_id"],
            "title": q_row["title"],
            "content": q_row["content"],
            "views": q_row["views"],
            "created_at": str(q_row["created_at"]) if q_row["created_at"] else None,
            "updated_at": str(q_row["updated_at"]) if q_row["updated_at"] else None,
            "author": {
                "username": q_row["username"],
                "full_name": q_row["full_name"],
            },
            "answers": answers,
        }
    finally:
        cursor.close()
        release_connection(conn)


def create_question(festival_name: str, user_id: int, title: str, content: str) -> int:
    """Create a new question"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute(
            """
            INSERT INTO questions (festival_name, user_id, title, content)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (festival_name, user_id, title, content),
        )
        result = cursor.fetchone()
        conn.commit()
        return result["id"]
    finally:
        cursor.close()
        release_connection(conn)


def update_question(question_id: int, title: str, content: str) -> bool:
    """Update a question"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute(
            """
            UPDATE questions
            SET title = %s, content = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (title, content, question_id),
        )
        conn.commit()
        rows_affected = cursor.rowcount
        return rows_affected > 0
    finally:
        cursor.close()
        release_connection(conn)


def delete_question(question_id: int) -> bool:
    """Delete a question"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute("DELETE FROM questions WHERE id = %s", (question_id,))
        conn.commit()
        rows_affected = cursor.rowcount
        return rows_affected > 0
    finally:
        cursor.close()
        release_connection(conn)


def create_answer(question_id: int, user_id: int, content: str) -> int:
    """Create a new answer"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute(
            """
            INSERT INTO answers (question_id, user_id, content)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (question_id, user_id, content),
        )
        result = cursor.fetchone()
        conn.commit()
        return result["id"]
    finally:
        cursor.close()
        release_connection(conn)


def update_answer(answer_id: int, content: str) -> bool:
    """Update an answer"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute(
            """
            UPDATE answers
            SET content = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (content, answer_id),
        )
        conn.commit()
        rows_affected = cursor.rowcount
        return rows_affected > 0
    finally:
        cursor.close()
        release_connection(conn)


def delete_answer(answer_id: int) -> bool:
    """Delete an answer"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute("DELETE FROM answers WHERE id = %s", (answer_id,))
        conn.commit()
        rows_affected = cursor.rowcount
        return rows_affected > 0
    finally:
        cursor.close()
        release_connection(conn)


def accept_answer(answer_id: int, question_id: int) -> bool:
    """Mark an answer as accepted"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        # First, unaccept all answers for this question
        cursor.execute(
            "UPDATE answers SET is_accepted = FALSE WHERE question_id = %s",
            (question_id,),
        )

        # Then accept the specified answer
        cursor.execute(
            "UPDATE answers SET is_accepted = TRUE WHERE id = %s AND question_id = %s",
            (answer_id, question_id),
        )
        conn.commit()
        rows_affected = cursor.rowcount
        return rows_affected > 0
    finally:
        cursor.close()
        release_connection(conn)


def get_user_questions(user_id: int, limit: int = 50) -> List[Dict]:
    """Get all questions by a user"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute(
            """
            SELECT
                q.id, q.festival_name, q.title, q.content, q.views,
                q.created_at, q.updated_at,
                (SELECT COUNT(*) FROM answers WHERE question_id = q.id) as answer_count
            FROM questions q
            WHERE q.user_id = %s
            ORDER BY q.created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()

        questions = []
        for row in rows:
            questions.append({
                "id": row["id"],
                "festival_name": row["festival_name"],
                "title": row["title"],
                "content": row["content"],
                "views": row["views"],
                "created_at": str(row["created_at"]) if row["created_at"] else None,
                "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
                "answer_count": row["answer_count"],
            })
        return questions
    finally:
        cursor.close()
        release_connection(conn)


def get_user_answers(user_id: int, limit: int = 50) -> List[Dict]:
    """Get all answers by a user"""
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute(
            """
            SELECT
                a.id, a.question_id, a.content, a.is_accepted,
                a.created_at, a.updated_at,
                q.title as question_title, q.festival_name
            FROM answers a
            JOIN questions q ON a.question_id = q.id
            WHERE a.user_id = %s
            ORDER BY a.created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()

        answers = []
        for row in rows:
            answers.append({
                "id": row["id"],
                "question_id": row["question_id"],
                "content": row["content"],
                "is_accepted": bool(row["is_accepted"]),
                "created_at": str(row["created_at"]) if row["created_at"] else None,
                "updated_at": str(row["updated_at"]) if row["updated_at"] else None,
                "question_title": row["question_title"],
                "festival_name": row["festival_name"],
            })
        return answers
    finally:
        cursor.close()
        release_connection(conn)
