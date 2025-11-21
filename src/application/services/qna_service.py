"""
Q&A service for question and answer management
"""

from typing import List, Dict, Optional
from fastapi import HTTPException
from src.infrastructure.persistence.database import get_db_connection


def get_questions_for_festival(
    festival_name: str,
    limit: int = 50,
    offset: int = 0
) -> List[Dict]:
    """Get all questions for a specific festival"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            q.id, q.festival_name, q.user_id, q.title, q.content, q.views,
            q.created_at, q.updated_at,
            u.username, u.full_name,
            (SELECT COUNT(*) FROM answers WHERE question_id = q.id) as answer_count
        FROM questions q
        JOIN users u ON q.user_id = u.id
        WHERE q.festival_name = ?
        ORDER BY q.created_at DESC
        LIMIT ? OFFSET ?
        """,
        (festival_name, limit, offset),
    )
    rows = cursor.fetchall()
    conn.close()

    questions = []
    for row in rows:
        questions.append({
            "id": row[0],
            "festival_name": row[1],
            "user_id": row[2],
            "title": row[3],
            "content": row[4],
            "views": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "author": {
                "username": row[8],
                "full_name": row[9],
            },
            "answer_count": row[10],
        })
    return questions


def get_question_by_id(question_id: int) -> Optional[Dict]:
    """Get a single question with its answers"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Increment view count
    cursor.execute(
        "UPDATE questions SET views = views + 1 WHERE id = ?",
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
        WHERE q.id = ?
        """,
        (question_id,),
    )
    q_row = cursor.fetchone()

    if not q_row:
        conn.close()
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
        WHERE a.question_id = ?
        ORDER BY a.is_accepted DESC, a.created_at ASC
        """,
        (question_id,),
    )
    answer_rows = cursor.fetchall()
    conn.close()

    answers = []
    for a_row in answer_rows:
        answers.append({
            "id": a_row[0],
            "question_id": a_row[1],
            "user_id": a_row[2],
            "content": a_row[3],
            "is_accepted": bool(a_row[4]),
            "created_at": a_row[5],
            "updated_at": a_row[6],
            "author": {
                "username": a_row[7],
                "full_name": a_row[8],
                "role": a_row[9],
            },
        })

    return {
        "id": q_row[0],
        "festival_name": q_row[1],
        "user_id": q_row[2],
        "title": q_row[3],
        "content": q_row[4],
        "views": q_row[5],
        "created_at": q_row[6],
        "updated_at": q_row[7],
        "author": {
            "username": q_row[8],
            "full_name": q_row[9],
        },
        "answers": answers,
    }


def create_question(festival_name: str, user_id: int, title: str, content: str) -> int:
    """Create a new question"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO questions (festival_name, user_id, title, content)
        VALUES (?, ?, ?, ?)
        """,
        (festival_name, user_id, title, content),
    )
    conn.commit()
    question_id = cursor.lastrowid
    conn.close()
    return question_id


def update_question(question_id: int, title: str, content: str) -> bool:
    """Update a question"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE questions
        SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (title, content, question_id),
    )
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


def delete_question(question_id: int) -> bool:
    """Delete a question"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM questions WHERE id = ?", (question_id,))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


def create_answer(question_id: int, user_id: int, content: str) -> int:
    """Create a new answer"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO answers (question_id, user_id, content)
        VALUES (?, ?, ?)
        """,
        (question_id, user_id, content),
    )
    conn.commit()
    answer_id = cursor.lastrowid
    conn.close()
    return answer_id


def update_answer(answer_id: int, content: str) -> bool:
    """Update an answer"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE answers
        SET content = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (content, answer_id),
    )
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


def delete_answer(answer_id: int) -> bool:
    """Delete an answer"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM answers WHERE id = ?", (answer_id,))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


def accept_answer(answer_id: int, question_id: int) -> bool:
    """Mark an answer as accepted"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # First, unaccept all answers for this question
    cursor.execute(
        "UPDATE answers SET is_accepted = 0 WHERE question_id = ?",
        (question_id,),
    )

    # Then accept the specified answer
    cursor.execute(
        "UPDATE answers SET is_accepted = 1 WHERE id = ? AND question_id = ?",
        (answer_id, question_id),
    )
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


def get_user_questions(user_id: int, limit: int = 50) -> List[Dict]:
    """Get all questions by a user"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            q.id, q.festival_name, q.title, q.content, q.views,
            q.created_at, q.updated_at,
            (SELECT COUNT(*) FROM answers WHERE question_id = q.id) as answer_count
        FROM questions q
        WHERE q.user_id = ?
        ORDER BY q.created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    questions = []
    for row in rows:
        questions.append({
            "id": row[0],
            "festival_name": row[1],
            "title": row[2],
            "content": row[3],
            "views": row[4],
            "created_at": row[5],
            "updated_at": row[6],
            "answer_count": row[7],
        })
    return questions


def get_user_answers(user_id: int, limit: int = 50) -> List[Dict]:
    """Get all answers by a user"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            a.id, a.question_id, a.content, a.is_accepted,
            a.created_at, a.updated_at,
            q.title as question_title, q.festival_name
        FROM answers a
        JOIN questions q ON a.question_id = q.id
        WHERE a.user_id = ?
        ORDER BY a.created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    answers = []
    for row in rows:
        answers.append({
            "id": row[0],
            "question_id": row[1],
            "content": row[2],
            "is_accepted": bool(row[3]),
            "created_at": row[4],
            "updated_at": row[5],
            "question_title": row[6],
            "festival_name": row[7],
        })
    return answers
